import os
import json
import re
import numpy
import requests
from datetime import datetime
from kubernetes.client import ApiException
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI
from flask import Flask, request, jsonify
from gevent.pywsgi import WSGIServer
from prometheus_client import Counter, Gauge, start_http_server
from kubernetes import client as k8s_client, config

# ===================== 全局基础配置 =====================
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# ========== LLM 配置 ==========
LLM_API_KEY = "sk-66f13eb900954ac0a6ef4768030df425"
LLM_BASE_URL = "https://api.deepseek.com"
llm_client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

# ========== 监控 & 集群地址 【根据你实际环境修改】 ==========
# 优先使用宿主机可直连的 Prometheus 地址
PROM_URL = "http://127.0.0.1:65175"
K8S_DEFAULT_NS = "sock-shop"

# 固定CPU查询语句
NODE_CPU_PROMQL = 'sum(rate(node_cpu_seconds_total{mode!="idle"}[2m])) by (instance) / 4 * 100'
POD_CPU_PROMQL_TPL = 'sum(rate(container_cpu_usage_seconds_total{{namespace="{ns}",pod="{pod}"}}[1m])) by (pod)'

# ========== 初始化 K8s 客户端 ==========
try:
    config.load_kube_config()
    k8s_v1 = k8s_client.CoreV1Api()
    print("✅ K8s 集群连接成功")
except Exception as e:
    print(f"❌ K8s 集群连接失败: {str(e)}")
    exit(1)

# ===================== 1. 运维知识库 =====================
knowledge_base = [
    {
        "name": "Pod 被销毁/异常退出",
        "content": "ChaosMesh 主动销毁 Pod、进程崩溃会导致微服务实例下线，前端出现503/访问超时，CPU、网络流量指标骤降，服务可用性下降。",
        "metrics": ["pod_status", "service_availability", "cpu_usage", "network_traffic"],
        "steps": [
            "1. 查看目标命名空间下所有Pod运行状态",
            "2. 检查Pod重启次数判断是否反复异常",
            "3. 读取Pod日志定位崩溃原因",
            "4. 验证K8s自愈能力，观察Pod是否自动重建"
        ]
    },
    {
        "name": "容器内存溢出(OOM)",
        "content": "容器内存使用超过K8s资源限制，系统强制杀死容器，触发Pod频繁重启，内存指标持续高位，服务间歇性不可用。",
        "metrics": ["memory_usage", "container_oom", "pod_restart_count"],
        "steps": [
            "1. 查看内存使用率是否超出阈值",
            "2. 核查Pod的CPU/内存资源配额配置",
            "3. 分析容器日志确认OOM报错",
            "4. 调整资源限制或优化应用内存泄漏问题"
        ]
    },
    {
        "name": "微服务CPU负载过高",
        "content": "流量突增、代码异常或ChaosMesh压力注入，导致Pod CPU使用率超标，接口响应变慢、整体吞吐量下降。",
        "metrics": ["cpu_usage", "request_latency", "throughput"],
        "steps": [
            "1. 定位高负载的微服务Pod",
            "2. 查询接口耗时与请求量指标",
            "3. 分析热点接口与异常请求",
            "4. 横向扩容实例或优化业务逻辑"
        ]
    },
    {
        "name": "集群网络故障/延迟",
        "content": "ChaosMesh注入网络延迟、丢包后，微服务间调用超时，页面加载缓慢，网络丢包、延迟指标异常。",
        "metrics": ["network_loss", "network_rtt", "request_timeout", "network_traffic"],
        "steps": [
            "1. 查看网络丢包率、延迟指标",
            "2. 追踪微服务调用链路",
            "3. 检查K8s网络策略配置",
            "4. 排查节点网络或负载均衡问题"
        ]
    },
    {
        "name": "数据库Pod异常",
        "content": "SockShop关联的MySQL/MongoDB数据库Pod故障，引发读写失败，购物车、订单等核心功能报错。",
        "metrics": ["db_connection_error", "disk_io", "pod_status"],
        "steps": [
            "1. 检查数据库Pod运行状态",
            "2. 查看磁盘IO与数据库连接数",
            "3. 分析数据库运行日志",
            "4. 重启异常数据库实例"
        ]
    }
]

# ===================== 2. BM25 知识检索模块 =====================
class BM25KnowledgeRetriever:
    def __init__(self, knowledge_list):
        self.knowledge_list = knowledge_list
        self.corpus = [k["metrics"] for k in knowledge_list]
        self.bm25 = BM25Okapi(self.corpus, k1=1.2, b=0.75)

    def retrieve(self, abnormal_metrics, top_k=2):
        scores = self.bm25.get_scores(abnormal_metrics)
        top_idx = numpy.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_idx:
            results.append({
                **self.knowledge_list[idx],
                "bm25_score": round(float(scores[idx]), 4)
            })
        return results

knowledge_retriever = BM25KnowledgeRetriever(knowledge_base)

# ===================== 3. 工具管理模块 =====================
class ToolManager:
    def __init__(self):
        self.tools = {}

    def register_tool(self, name, description, parameters, execute_func):
        self.tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "execute": execute_func
        }

    def execute_tool(self, tool_name, **kwargs):
        if tool_name not in self.tools:
            return f"错误：工具 {tool_name} 不存在"
        try:
            return self.tools[tool_name]["execute"](**kwargs)
        except Exception as e:
            return f"工具执行失败：{str(e)}"

    def get_all_tool_descriptions(self):
        return [f"工具名：{t['name']}；功能：{t['description']}" for t in self.tools.values()]

# ---------------------- 通用Prometheus查询函数 ----------------------
def prom_query(promql, timeout=8):
    api_url = f"{PROM_URL}/api/v1/query"
    try:
        resp = requests.get(api_url, params={"query": promql}, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# ---------------------- 工具1：查询K8s Pod 状态 ----------------------
def get_k8s_pod_status(namespace=K8S_DEFAULT_NS, pod_name=""):
    try:
        if pod_name:
            pod = k8s_v1.read_namespaced_pod(name=pod_name, namespace=namespace)
            status = pod.status.phase
            restart_count = sum(c.restart_count for c in pod.status.container_statuses) if pod.status.container_statuses else 0
            res = f"命名：{namespace}，Pod：{pod_name}，状态：{status}，重启次数：{restart_count}"
        else:
            pod_list = k8s_v1.list_namespaced_pod(namespace=namespace)
            abnormal_pods = []
            for pod in pod_list.items:
                phase = pod.status.phase
                restart = sum(c.restart_count for c in pod.status.container_statuses) if pod.status.container_statuses else 0
                if phase != "Running" or restart > 2:
                    abnormal_pods.append(f"Pod:{pod.metadata.name} 状态:{phase} 重启:{restart}")
            res = f"命名空间 {namespace} 异常Pod：{'; '.join(abnormal_pods)}" if abnormal_pods else f"命名空间 {namespace} 所有Pod运行正常"
        return json.dumps({"pod_info": res}, ensure_ascii=False)
    except ApiException as e:
        return json.dumps({"error": f"K8s API 异常: {e.reason}"}, ensure_ascii=False)

# ---------------------- 工具2：查询Prometheus指标 ----------------------
def query_prom_metric(metric_name, start_time="", end_time=""):
    metric_expr_map = {
        "node_cpu_usage": NODE_CPU_PROMQL,
        "cpu_usage": f'sum(rate(container_cpu_usage_seconds_total{{namespace="{K8S_DEFAULT_NS}"}}[1m])) by (pod)',
        "memory_usage": f'sum(container_memory_usage_bytes{{namespace="{K8S_DEFAULT_NS}"}}) / 1024 / 1024 / 1024',
        "pod_restart_count": f'sum(kube_pod_container_status_restarts_total{{namespace="{K8S_DEFAULT_NS}"}})',
        "network_loss": f'sum(rate(container_network_receive_errors_total{{namespace="{K8S_DEFAULT_NS}"}}[1m]))'
    }
    expr = metric_expr_map.get(metric_name, "")
    if not expr:
        return json.dumps({"result": f"无对应PromQL: {metric_name}"}, ensure_ascii=False)

    data = prom_query(expr)
    if data.get("status") != "success":
        return json.dumps({"error": data.get("msg", "Prometheus查询失败")}, ensure_ascii=False)

    result_list = data.get("data", {}).get("result", [])
    output = []
    for item in result_list:
        metric = item["metric"]
        val = round(float(item["value"][1]), 2)
        if metric_name == "node_cpu_usage":
            output.append(f"节点 {metric.get('instance','unknown')} CPU使用率: {val}%")
        elif metric_name == "memory_usage":
            output.append(f"Pod {metric.get('pod','global')} 内存: {val} GB")
        else:
            output.append(f"Pod {metric.get('pod','global')} 指标值: {val}")
    output = output if output else ["未查询到监控数据"]
    return json.dumps({"metric": metric_name, "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "result": "; ".join(output)}, ensure_ascii=False)

# ---------------------- 工具3：查询Pod日志 ----------------------
def get_pod_logs(namespace=K8S_DEFAULT_NS, pod_name="", tail_lines=50):
    if not pod_name:
        return json.dumps({"error": "必须指定Pod名称"}, ensure_ascii=False)
    try:
        log_data = k8s_v1.read_namespaced_pod_log(name=pod_name, namespace=namespace, tail_lines=tail_lines, timestamps=False)
        error_lines = [line for line in log_data.splitlines() if "ERROR" in line or "OOM" in line or "crash" in line]
        log_res = "异常日志：\n" + "\n".join(error_lines[-10:]) if error_lines else "Pod日志无明显报错"
        return json.dumps({"namespace": namespace, "pod_name": pod_name, "logs": log_res}, ensure_ascii=False)
    except ApiException as e:
        return json.dumps({"error": f"读取日志失败: {e.reason}"}, ensure_ascii=False)

# ---------------------- 工具4：CPU状态综合检测 ----------------------
def get_cpu_status(namespace=K8S_DEFAULT_NS, pod_name="", threshold=0.8):
    node_data = prom_query(NODE_CPU_PROMQL)
    node_cpu_text = "节点CPU数据查询失败"
    if node_data.get("status") == "success" and node_data["data"]["result"]:
        item = node_data["data"]["result"][0]
        node_cpu_text = f"节点 {item['metric']['instance']} 实时CPU使用率: {round(float(item['value'][1]),2)}%"

    if pod_name:
        pod_expr = POD_CPU_PROMQL_TPL.format(ns=namespace, pod=pod_name)
    else:
        pod_expr = f'sum(rate(container_cpu_usage_seconds_total{{namespace="{namespace}"}}[1m])) by (pod)'
    pod_data = prom_query(pod_expr)

    if pod_data.get("status") != "success" or not pod_data["data"]["result"]:
        return json.dumps({
            "status": "no_data",
            "message": "未查询到Pod CPU数据",
            "node_cpu": node_cpu_text
        }, ensure_ascii=False)

    cpu_status = []
    overall_status = "normal"
    for item in pod_data["data"]["result"]:
        pod = item["metric"]["pod"]
        cpu_cores = round(float(item["value"][1]), 3)
        if cpu_cores > threshold * 1.5:
            state = "critical"
            overall_status = "critical"
        elif cpu_cores > threshold:
            state = "warning"
            overall_status = "warning"
        else:
            state = "normal"
        cpu_status.append({"pod": pod, "cpu_cores": cpu_cores, "threshold": threshold, "state": state})

    return json.dumps({
        "namespace": namespace,
        "overall_status": overall_status,
        "threshold": threshold,
        "pod_cpu_details": cpu_status,
        "node_cpu_usage": node_cpu_text
    }, ensure_ascii=False)

# 注册所有工具
tool_manager = ToolManager()
tool_manager.register_tool("get_k8s_pod_status", "查询K8s Pod运行状态、重启次数",
                           [{"name":"namespace","type":"str"},{"name":"pod_name","type":"str"}], get_k8s_pod_status)
tool_manager.register_tool("query_prom_metric", "查询Prometheus监控指标",
                           [{"name":"metric_name","type":"str"},{"name":"start_time","type":"str"},{"name":"end_time","type":"str"}], query_prom_metric)
tool_manager.register_tool("get_pod_logs", "读取Pod运行日志",
                           [{"name":"namespace","type":"str"},{"name":"pod_name","type":"str"},{"name":"tail_lines","type":"int"}], get_pod_logs)
tool_manager.register_tool("get_cpu_status", "检测Pod与节点CPU实时状态",
                           [{"name":"namespace","type":"str"},{"name":"pod_name","type":"str"},{"name":"threshold","type":"float"}], get_cpu_status)

# ===================== 4. 工具语义匹配模块 【修复数组维度BUG】 =====================
class ToolMatcher:
    def __init__(self, tool_manager):
        self.tool_manager = tool_manager
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        tool_descs = tool_manager.get_all_tool_descriptions()
        self.tool_embeddings = self.model.encode(tool_descs)
        self.tool_names = list(tool_manager.tools.keys())

    def match(self, context, top_k=2):
        context_emb = self.model.encode([context])
        # 修复维度问题：保证都是二维数组
        sims = cosine_similarity(context_emb, self.tool_embeddings)[0]
        top_idx = numpy.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_idx:
            tool = self.tool_manager.tools[self.tool_names[idx]]
            results.append({
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"],
                "similarity": round(float(sims[idx]), 4)
            })
        return results

tool_matcher = ToolMatcher(tool_manager)

# ===================== 5. LLM 对话 =====================
def llm_chat(prompt, temperature=0):
    response = llm_client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# ===================== 6. Prompt 构建 =====================
def build_diagnosis_prompt(anomaly_info, retrieved_knowledge, matched_tools):
    knowledge_text = ""
    for idx, k in enumerate(retrieved_knowledge):
        knowledge_text += f"\n【知识{idx+1}：{k['name']}】\n原理：{k['content']}\n关联指标：{', '.join(k['metrics'])}\n排查步骤：{chr(10).join(k['steps'])}"
    tool_text = ""
    for idx, t in enumerate(matched_tools):
        param_str = ", ".join([f"{p['name']}({p['type']})" for p in t['parameters']])
        tool_text += f"\n【工具{idx+1}：{t['name']}】\n功能：{t['description']}\n参数：{param_str}"
    prompt = f"""你是资深云原生运维专家。
【异常告警】{anomaly_info}
【参考知识】{knowledge_text}
【可用工具】{tool_text}
严格按格式输出：
1. Thought：思考过程
2. Action：工具名称
3. Arguments：JSON格式入参"""
    return prompt

# ===================== 7. 核心诊断函数 =====================
def run_diagnosis(anomaly_desc, abnormal_metrics):
    print("\n🔍 【主动实时查询节点CPU】")
    node_cpu_res = query_prom_metric("node_cpu_usage")
    node_cpu_data = json.loads(node_cpu_res)
    real_cpu_value = node_cpu_data.get("result", "无数据")
    print(f"✅ 当前节点CPU实时值：{real_cpu_value}")

    anomaly_desc += f"\n【实时节点CPU使用率】：{real_cpu_value}"

    retrieved_knowledge = knowledge_retriever.retrieve(abnormal_metrics, top_k=2)
    matched_tools = tool_matcher.match(anomaly_desc, top_k=2)
    prompt = build_diagnosis_prompt(anomaly_desc, retrieved_knowledge, matched_tools)
    llm_output = llm_chat(prompt)

    action_match = re.search(r"Action[:：]\s*(\w+)", llm_output)
    args_match = re.search(r"Arguments[:：]\s*(\{.*?\})", llm_output, re.DOTALL)
    if not action_match:
        return {"status": "failed", "reason": "LLM未生成有效工具调用"}

    tool_name = action_match.group(1)
    try:
        args = json.loads(args_match.group(1)) if args_match else {}
    except:
        args = {}

    tool_result = tool_manager.execute_tool(tool_name, **args)
    final_prompt = f"""结合运维数据输出诊断报告（200字内，根因+解决方案）：
异常：{anomaly_desc}
工具：{tool_name}
返回数据：{tool_result}"""
    conclusion = llm_chat(final_prompt)

    return {
        "status": "success",
        "anomaly_desc": anomaly_desc,
        "abnormal_metrics": abnormal_metrics,
        "retrieved_knowledge": retrieved_knowledge,
        "matched_tools": matched_tools,
        "tool_call": {"name": tool_name, "args": args},
        "tool_result": tool_result,
        "diagnosis_conclusion": conclusion,
        "real_cpu_value": real_cpu_value
    }

# ===================== 8. Flask Webhook =====================
DIAGNOSIS_TOTAL = Counter("db_diagnosis_total", "诊断任务总数")
DIAGNOSIS_SUCCESS = Counter("db_diagnosis_success", "诊断成功数")
DIAGNOSIS_FAILED = Counter("db_diagnosis_failed", "诊断失败数")
DIAGNOSIS_RUNNING = Gauge("db_diagnosis_running", "当前运行任务数")

app = Flask(__name__)

def parse_prometheus_alert(alert_json):
    alerts = alert_json.get("alerts", [])
    if not alerts:
        return None, []
    alert = alerts[0]
    status = alert.get("status", "firing")
    labels = alert.get("labels", {})
    annotations = alert.get("annotations", {})
    summary = annotations.get("summary", "未知异常")
    desc = annotations.get("description", "")
    anomaly_desc = f"【告警状态】{status} | 标题：{summary} | 原始告警快照：{desc}"

    metric_map = {
        "cpu_high":"cpu_usage","memory_high":"memory_usage",
        "pod_abnormal":"pod_status","pod_restart":"pod_restart_count",
        "network_error":"network_loss","db_failure":"db_connection_error"
    }
    abnormal_metrics = []
    for key in labels:
        if key in metric_map:
            abnormal_metrics.append(metric_map[key])
    if "CPU" in summary:
        abnormal_metrics.append("cpu_usage")
    if "内存" in summary:
        abnormal_metrics.append("memory_usage")
    abnormal_metrics = list(set(abnormal_metrics))
    return anomaly_desc, abnormal_metrics

@app.route("/prometheus_webhook", methods=["POST"])
def prometheus_webhook():
    DIAGNOSIS_TOTAL.inc()
    DIAGNOSIS_RUNNING.inc()
    try:
        alert_data = request.get_json()
        print(f"\n===== 收到Prometheus告警报文 =====")

        anomaly_desc, abnormal_metrics = parse_prometheus_alert(alert_data)
        if not anomaly_desc or not abnormal_metrics:
            DIAGNOSIS_FAILED.inc()
            DIAGNOSIS_RUNNING.dec()
            return jsonify({"code": 400, "msg": "无有效异常指标"}), 400

        diag_result = run_diagnosis(anomaly_desc, abnormal_metrics)
        if diag_result["status"] == "success":
            DIAGNOSIS_SUCCESS.inc()
            print(f"\n===== 诊断完成 =====")
            print(f"📊 告警快照旧值：{anomaly_desc}")
            print(f"📊 代码实时查询值：{diag_result['real_cpu_value']}")
            print(f"💡 诊断结论：{diag_result['diagnosis_conclusion']}")
        else:
            DIAGNOSIS_FAILED.inc()
            print(f"\n===== 诊断失败 =====")
            print(f"原因：{diag_result['reason']}")

        DIAGNOSIS_RUNNING.dec()
        return jsonify({"code": 200, "msg": "诊断完成", "data": diag_result})
    except Exception as e:
        DIAGNOSIS_FAILED.inc()
        DIAGNOSIS_RUNNING.dec()
        print(f"\n===== 服务异常 =====")
        print(f"错误：{str(e)}")
        DIAGNOSIS_RUNNING.dec()
        return jsonify({"code": 500, "msg": f"服务异常：{str(e)}"}), 500

# ===================== 启动服务 =====================
if __name__ == "__main__":
    start_http_server(5001)
    print("✅ 自监控指标：http://127.0.0.1:5001/metrics")
    print("✅ Webhook 地址：http://0.0.0.0:5000/prometheus_webhook")
    print("✅ 服务启动，每次告警都会实时查询CPU")

    http_server = WSGIServer(("0.0.0.0", 5000), app)
    http_server.serve_forever()