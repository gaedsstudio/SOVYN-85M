"""
AI 벤치마크 - 추론/수학/코딩/논리 능력 테스트
사용법:
  python benchmark.py              # 자체 모델 벤치마크
  python benchmark.py --show       # 문제만 보기
  python benchmark.py --export     # JSON으로 내보내기
"""
import torch
import time
import json
import sys
import re
import math
from pathlib import Path
from fractions import Fraction

# ─── 벤치마크 문제 정의 ───────────────────────────────────

BENCHMARKS = {
    "산술_기본": [
        {"q": "문제: 7 + 8을 계산하시오.", "answer": "15"},
        {"q": "문제: 25 - 13을 계산하시오.", "answer": "12"},
        {"q": "문제: 6 * 9를 계산하시오.", "answer": "54"},
        {"q": "문제: 100 - 37을 계산하시오.", "answer": "63"},
        {"q": "문제: 12 * 11을 계산하시오.", "answer": "132"},
    ],
    "연산_우선순위": [
        {"q": "문제: 3 + 5 * 2를 계산하시오.", "answer": "13"},
        {"q": "문제: 10 - 2 * 3를 계산하시오.", "answer": "4"},
        {"q": "문제: 4 * 3 + 7를 계산하시오.", "answer": "19"},
        {"q": "문제: 20 + 6 / 2를 계산하시오.", "answer": "23"},
        {"q": "문제: 8 * 5 - 10를 계산하시오.", "answer": "30"},
        {"q": "문제: 2 + 3 * 4 - 1를 계산하시오.", "answer": "13"},
        {"q": "문제: 100 - 10 * 5를 계산하시오.", "answer": "50"},
        {"q": "문제: 7 * 8 + 3 * 2를 계산하시오.", "answer": "62"},
    ],
    "괄호_연산": [
        {"q": "문제: (3 + 5) * 2를 계산하시오.", "answer": "16"},
        {"q": "문제: (10 - 4) * (3 + 2)를 계산하시오.", "answer": "30"},
        {"q": "문제: (8 + 2) * 5를 계산하시오.", "answer": "50"},
        {"q": "문제: 3 * (7 - 4)를 계산하시오.", "answer": "9"},
        {"q": "문제: (15 + 5) * (2 - 1)를 계산하시오.", "answer": "20"},
    ],
    "방정식": [
        {"q": "문제: 2x + 4 = 10일 때, x의 값을 구하시오.", "answer": "3"},
        {"q": "문제: 5x - 15 = 0일 때, x의 값을 구하시오.", "answer": "3"},
        {"q": "문제: 3x + 7 = 22일 때, x의 값을 구하시오.", "answer": "5"},
        {"q": "문제: x + 10 = 25일 때, x의 값을 구하시오.", "answer": "15"},
        {"q": "문제: 4x - 8 = 12일 때, x의 값을 구하시오.", "answer": "5"},
    ],
    "리스트_연산": [
        {"q": "문제: lst = [5, 3, 8, 1, 9]일 때, max(lst)의 값은?", "answer": "9"},
        {"q": "문제: lst = [5, 3, 8, 1, 9]일 때, min(lst)의 값은?", "answer": "1"},
        {"q": "문제: lst = [5, 3, 8, 1, 9]일 때, sum(lst)의 값은?", "answer": "26"},
        {"q": "문제: lst = [5, 3, 8, 1, 9]일 때, len(lst)의 값은?", "answer": "5"},
        {"q": "문제: lst = [10, 20, 30, 40]일 때, lst[2]의 값은?", "answer": "30"},
        {"q": "문제: lst = [7, 2, 9, 4, 1]일 때, sorted(lst)의 결과는?", "answer": "[1, 2, 4, 7, 9]"},
    ],
    "코드_트레이싱": [
        {"q": "문제: 다음 코드의 출력은?\nresult = 0\nfor i in range(1, 6):\n    result += i\nprint(result)", "answer": "15"},
        {"q": "문제: 다음 코드의 출력은?\nresult = 1\nfor i in range(1, 5):\n    result *= i\nprint(result)", "answer": "24"},
        {"q": "문제: 다음 코드의 출력은?\nx = 10\nif x > 5:\n    print('크다')\nelse:\n    print('작다')", "answer": "크다"},
        {"q": "문제: 다음 코드의 출력은?\nx = 7\nif x % 2 == 0:\n    print('짝수')\nelse:\n    print('홀수')", "answer": "홀수"},
        {"q": "문제: 다음 코드의 출력은?\na = 5\nb = 3\na, b = b, a\nprint(a, b)", "answer": "3 5"},
    ],
    "논리": [
        {"q": '문제: P = 참, Q = 거짓일 때, P AND Q의 값은?', "answer": "거짓"},
        {"q": '문제: P = 참, Q = 거짓일 때, P OR Q의 값은?', "answer": "참"},
        {"q": '문제: P = 참일 때, NOT P의 값은?', "answer": "거짓"},
        {"q": '문제: P = 거짓, Q = 참일 때, P AND Q의 값은?', "answer": "거짓"},
        {"q": '문제: P = 참, Q = 참일 때, P AND Q의 값은?', "answer": "참"},
    ],
    "숫자_성질": [
        {"q": "문제: 456의 각 자릿수의 합을 구하시오.", "answer": "15"},
        {"q": "문제: 17은 소수인가?", "answer": "소수"},
        {"q": "문제: 24는 짝수인가 홀수인가?", "answer": "짝수"},
        {"q": "문제: 37을 5으로 나눈 나머지를 구하시오.", "answer": "2"},
        {"q": "문제: 12와 18의 최대공약수를 구하시오.", "answer": "6"},
    ],
    "서술형": [
        {"q": "문제: 현재 철수는 10살이고 영희는 15살이다. 3년 후 두 사람의 나이 합은?", "answer": "31살"},
        {"q": "문제: 500원짜리 물건을 3개 사고 2000원을 냈다. 거스름돈은?", "answer": "500원"},
        {"q": "문제: 200의 25%는 얼마인가?", "answer": "50"},
        {"q": "문제: 시속 60km로 3시간 이동하면 거리는?", "answer": "180km"},
        {"q": "문제: 가로 8, 세로 5인 직사각형의 넓이를 구하시오.", "answer": "40"},
    ],
    "수열": [
        {"q": "문제: 2, 5, 8, 11, 14, ? 의 다음 수를 구하시오.", "answer": "17"},
        {"q": "문제: 1, 2, 4, 8, 16, ? 의 다음 수를 구하시오.", "answer": "32"},
        {"q": "문제: 1, 1, 2, 3, 5, 8, ? 의 다음 수를 구하시오.", "answer": "13"},
    ],
}

TOTAL_PROBLEMS = sum(len(v) for v in BENCHMARKS.values())


def extract_answer(text, expected):
    """모델 출력에서 답 추출"""
    # "답: XXX" 패턴
    m = re.search(r'답[:\s]+(.+?)(?:\n|$)', text)
    if m:
        return m.group(1).strip()

    # 마지막 숫자/단어
    lines = text.strip().split('\n')
    for line in reversed(lines):
        line = line.strip()
        if line and not line.startswith('풀이') and not line.startswith('문제'):
            # 숫자 추출
            nums = re.findall(r'-?\d+\.?\d*', line)
            if nums:
                return nums[-1]
            return line
    return text.strip()


def check_answer(predicted, expected):
    """답이 맞는지 확인 (유연한 비교)"""
    pred = predicted.strip().rstrip('.').strip()
    exp = expected.strip()

    # 정확히 일치
    if pred == exp:
        return True

    # 숫자 비교
    try:
        pred_nums = re.findall(r'-?\d+\.?\d*', pred)
        exp_nums = re.findall(r'-?\d+\.?\d*', exp)
        if pred_nums and exp_nums:
            if float(pred_nums[-1]) == float(exp_nums[-1]):
                return True
    except (ValueError, IndexError):
        pass

    # 문자열 포함 비교 (짝수/홀수, 소수 등)
    if exp in pred:
        return True

    # 리스트 비교
    if exp.startswith('[') and pred.startswith('['):
        try:
            if eval(pred) == eval(exp):
                return True
        except:
            pass

    return False


def load_model():
    """모델 로드"""
    sys.path.insert(0, str(Path(__file__).parent))
    from train_125m import BPETokenizer, GPT125M, ModelConfig

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = ModelConfig()

    tok_path = Path(cfg.model_dir) / "tokenizer.json"
    model_path = Path(cfg.model_dir) / "final.pt"
    if not model_path.exists():
        model_path = Path(cfg.model_dir) / "best.pt"

    if not tok_path.exists() or not model_path.exists():
        print("  ❌ 학습된 모델이 없습니다!")
        return None, None, None, None

    tokenizer = BPETokenizer()
    tokenizer.load(str(tok_path))

    ckpt = torch.load(str(model_path), weights_only=True, map_location=device)
    saved = ckpt.get('config', {})
    for k, v in saved.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.vocab_size = tokenizer.vocab_size

    model = GPT125M(cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    step = ckpt.get('step', '?')
    val_loss = ckpt.get('val_loss', '?')
    print(f"  모델 로드: {model.num_params/1e6:.1f}M params, step={step}, val_loss={val_loss}")

    return model, tokenizer, device, cfg


def run_benchmark(model, tokenizer, device, verbose=True):
    """전체 벤치마크 실행"""
    print("\n" + "=" * 60)
    print("  AI 벤치마크 시작")
    print("=" * 60)

    results = {}
    total_correct = 0
    total_count = 0
    total_time = 0

    for category, problems in BENCHMARKS.items():
        cat_correct = 0
        cat_results = []

        if verbose:
            print(f"\n{'─' * 50}")
            print(f"  [{category}] ({len(problems)}문제)")
            print(f"{'─' * 50}")

        for i, prob in enumerate(problems):
            question = prob['q']
            expected = prob['answer']

            # 모델 추론
            tok_ids = tokenizer.encode(question)
            idx = torch.tensor([tok_ids], device=device)

            t0 = time.time()
            with torch.no_grad():
                out = model.generate(idx, max_new_tokens=200, temperature=0.3, top_k=20)
            dt = time.time() - t0
            total_time += dt

            response = tokenizer.decode(out[0].tolist())
            if response.startswith(question):
                response = response[len(question):]

            predicted = extract_answer(response, expected)
            correct = check_answer(predicted, expected)

            if correct:
                cat_correct += 1
                total_correct += 1
            total_count += 1

            status = "✅" if correct else "❌"
            cat_results.append({
                "question": question.split('\n')[0][:60],
                "expected": expected,
                "predicted": predicted,
                "correct": correct,
                "time": round(dt, 2),
            })

            if verbose:
                q_short = question.split('\n')[0][:50]
                print(f"  {status} {q_short}...")
                print(f"     정답: {expected} | 예측: {predicted} ({dt:.1f}s)")

        accuracy = cat_correct / len(problems) * 100 if problems else 0
        results[category] = {
            "correct": cat_correct,
            "total": len(problems),
            "accuracy": accuracy,
            "details": cat_results,
        }

        if verbose:
            print(f"  ➜ {category}: {cat_correct}/{len(problems)} ({accuracy:.0f}%)")

    # ─── 최종 결과 ───
    overall = total_correct / total_count * 100 if total_count else 0
    print("\n" + "═" * 60)
    print("  벤치마크 결과")
    print("═" * 60)
    print(f"\n  {'카테고리':<20} {'점수':>10} {'정확도':>10}")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    for cat, res in results.items():
        bar = "█" * int(res['accuracy'] / 10) + "░" * (10 - int(res['accuracy'] / 10))
        print(f"  {cat:<20} {res['correct']:>3}/{res['total']:<5}  {bar} {res['accuracy']:.0f}%")
    print(f"  {'─'*20} {'─'*10} {'─'*10}")
    print(f"  {'전체':<20} {total_correct:>3}/{total_count:<5}  {'█' * int(overall/10)}{'░' * (10-int(overall/10))} {overall:.1f}%")
    print(f"\n  총 소요시간: {total_time:.1f}초")
    print(f"  문제당 평균: {total_time/total_count:.2f}초")

    # 등급
    if overall >= 90:
        grade = "S (탁월)"
    elif overall >= 80:
        grade = "A (우수)"
    elif overall >= 70:
        grade = "B (양호)"
    elif overall >= 60:
        grade = "C (보통)"
    elif overall >= 40:
        grade = "D (미흡)"
    else:
        grade = "F (부족)"
    print(f"\n  종합 등급: {grade}")
    print("═" * 60)

    return {
        "total_correct": total_correct,
        "total_count": total_count,
        "overall_accuracy": round(overall, 1),
        "grade": grade,
        "total_time": round(total_time, 1),
        "categories": results,
    }


def show_problems():
    """문제 목록만 출력"""
    print("\n  벤치마크 문제 목록")
    print("=" * 60)
    idx = 1
    for cat, problems in BENCHMARKS.items():
        print(f"\n  [{cat}]")
        for p in problems:
            q = p['q'].split('\n')[0][:60]
            print(f"    {idx:2}. {q}  →  {p['answer']}")
            idx += 1
    print(f"\n  총 {TOTAL_PROBLEMS}문제")


def export_benchmark():
    """벤치마크 문제를 JSON으로 저장"""
    data = []
    for cat, problems in BENCHMARKS.items():
        for p in problems:
            data.append({"category": cat, **p})
    with open("benchmark_problems.json", "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"  benchmark_problems.json 저장 ({len(data)}문제)")


def main():
    print("═" * 60)
    print("  AI 벤치마크 v1.0")
    print(f"  카테고리: {len(BENCHMARKS)}개 | 문제: {TOTAL_PROBLEMS}개")
    print("═" * 60)

    if "--show" in sys.argv:
        show_problems()
        return

    if "--export" in sys.argv:
        export_benchmark()
        return

    model, tokenizer, device, cfg = load_model()
    if model is None:
        return

    results = run_benchmark(model, tokenizer, device)

    # 결과 저장
    out_path = "benchmark_result.json"
    with open(out_path, "w", encoding="utf-8") as f:
        # details 안의 큰 텍스트는 제외
        save_data = {
            "total_correct": results["total_correct"],
            "total_count": results["total_count"],
            "overall_accuracy": results["overall_accuracy"],
            "grade": results["grade"],
            "total_time": results["total_time"],
            "categories": {
                k: {"correct": v["correct"], "total": v["total"], "accuracy": v["accuracy"]}
                for k, v in results["categories"].items()
            }
        }
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n  결과 저장: {out_path}")


if __name__ == "__main__":
    main()
