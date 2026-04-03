import json
import requests
from tqdm import tqdm
import argparse
from pathlib import Path

def load_test_cases(file_path: str = "test/test_cases.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def evaluate(api_url: str = "http://localhost:8000/chat", test_file: str = "test/test_cases.json"):
    test_cases = load_test_cases(test_file)
    correct = 0
    results = []

    print("🚀 开始RAG专业评估（工业售后服务标准测试集）...\n")
    for case in tqdm(test_cases, desc="评估中"):
        try:
            # 你的FastAPI接口是 /chat，payload 就是 {"question": "..."}
            resp = requests.post(api_url, json={"question": case["question"]}, timeout=15)
            if resp.status_code == 200:
                answer = resp.json().get("answer", "无返回")
            else:
                answer = f"HTTP错误 {resp.status_code}"
        except Exception as e:
            answer = f"请求异常: {e}"

        # 人工快速判断（控制台输入 y/n）
        print(f"\n📌 问题: {case['question']}")
        print(f"🤖 系统回答: {answer[:400]}..." if len(answer) > 400 else answer)
        is_correct = input("✅ 这个回答是否正确、专业且基于PDF文档？(y/n): ").strip().lower()

        results.append({
            "question": case["question"],
            "answer": answer,
            "correct": is_correct == "y"
        })
        if is_correct == "y":
            correct += 1

    accuracy = (correct / len(test_cases)) * 100
    print(f"\n🎯 【评估报告】")
    print(f"端到端回答准确率: {accuracy:.1f}% （{correct}/{len(test_cases)}条）")
    print(f"测试集规模: 12条工业售后服务国家标准真实问题")

    # 保存报告（后面可以截图放GitHub）
    report_path = Path("test/evaluation_report.json")
    report_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 完整报告已保存到: test/evaluation_report.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG准确率评估工具")
    parser.add_argument("--api", default="http://localhost:8000/chat", help="FastAPI地址")
    args = parser.parse_args()
    evaluate(args.api)