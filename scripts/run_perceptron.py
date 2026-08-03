import sys
from pathlib import Path
import asyncio

# Add backend folder to Python path
BACKEND_DIR = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from app.benchmarks.runner import BenchmarkRunner
from app.benchmarks.challenges.network import build_perceptron


async def main():
    challenge = build_perceptron()

    runner = BenchmarkRunner(
        verbose=True,
        max_iterations=10
    )

    result = await runner.run_challenge(challenge)

    print("\n===== RESULT =====")
    print("Passed:", result.passed)
    print("Flag found:", result.flag_found)
    print("Answer:")
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())