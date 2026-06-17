import argparse
import time

import requests


QUESTIONS = [
    "What services are covered by the AWS Customer Agreement?",
    "Who is responsible for end users under the agreement?",
    "What are the customer payment obligations?",
    "When can AWS suspend services?",
    "What happens when the agreement is terminated?",
    "How does AWS define service offerings?",
    "What does the agreement say about taxes?",
    "Can customers resell AWS services?",
    "What are the acceptable use obligations?",
    "How are disputes handled?",
    "What limitation of liability applies?",
    "What does the agreement say about confidential information?",
    "Who owns customer content?",
    "What security obligations does AWS describe?",
    "Can AWS change service terms?",
    "What law governs the agreement?",
    "How are notices sent?",
    "What is the term of the agreement?",
    "What happens to customer content after termination?",
    "What does the agreement say about fees?",
    "Does the document mention Kubernetes pod autoscaling?",
    "What is the capital of Brazil?",
    "Does the agreement explain how to bake sourdough bread?",
    "Who won the 2022 FIFA World Cup?",
    "Does the document include medical diagnosis advice?",
    "Can I mine cryptocurrency under this agreement?",
    "What are AWS trademarks rules?",
    "What indemnification obligations exist?",
    "What are customer responsibilities for passwords?",
    "Does AWS provide warranties in this agreement?",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000")
    args = parser.parse_args()

    requests.post(f"{args.api}/ingest", timeout=180).raise_for_status()
    for question in QUESTIONS:
        response = requests.post(f"{args.api}/ask", json={"query": question}, timeout=120)
        print(response.status_code, question)
        time.sleep(0.2)


if __name__ == "__main__":
    main()
