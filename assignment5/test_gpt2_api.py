"""
Assignment 5: Test script for Fine-tuned GPT-2 API endpoints
Tests the /gpt2/answer and /gpt2/answer/multiple endpoints
"""

import requests
import json
from typing import Dict, Any


BASE_URL = "http://localhost:8000"


def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_api_health():
    """Test if API is running"""
    print_section("Testing API Health")
    
    try:
        response = requests.get(f"{BASE_URL}/")
        response.raise_for_status()
        data = response.json()
        
        print(f"✅ API Status: {data['status']}")
        print(f"📋 Message: {data['message']}")
        print(f"🔗 Available Endpoints:")
        for endpoint in data['endpoints']:
            print(f"   - {endpoint}")
        
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ API Health Check Failed: {e}")
        return False


def test_gpt2_single_answer():
    """Test single answer generation"""
    print_section("Testing Single Answer Generation (/gpt2/answer)")
    
    test_questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "When was Python programming language created?",
        "Where is the Eiffel Tower located?",
        "How does photosynthesis work?",
    ]
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n[Test {i}/{len(test_questions)}]")
        print(f"📝 Question: {question}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/gpt2/answer",
                json={
                    "question": question,
                    "max_length": 150,
                    "temperature": 0.7,
                    "top_p": 0.9
                },
                timeout=30
            )
            
            response.raise_for_status()
            data = response.json()
            
            print(f"💬 Answer: {data['answer']}")
            print(f"🤖 Model: {data['model']}")
            
            # Validate format
            answer = data['answer']
            if "That is a great question" in answer:
                print("✅ Format check: Contains opening phrase")
            else:
                print("⚠️  Format check: Missing opening phrase")
            
            if "Let me know if you have any other questions" in answer:
                print("✅ Format check: Contains closing phrase")
            else:
                print("⚠️  Format check: Missing closing phrase")
            
        except requests.exceptions.Timeout:
            print("⏰ Request timed out (model might be loading or generating)")
        except requests.exceptions.RequestException as e:
            print(f"❌ Request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Error details: {e.response.text}")


def test_gpt2_multiple_answers():
    """Test multiple answer generation"""
    print_section("Testing Multiple Answer Generation (/gpt2/answer/multiple)")
    
    question = "What is artificial intelligence?"
    num_responses = 3
    
    print(f"📝 Question: {question}")
    print(f"🔢 Requested responses: {num_responses}")
    
    try:
        response = requests.post(
            f"{BASE_URL}/gpt2/answer/multiple",
            json={
                "question": question,
                "num_responses": num_responses,
                "max_length": 150,
                "temperature": 0.8
            },
            timeout=60
        )
        
        response.raise_for_status()
        data = response.json()
        
        print(f"\n🤖 Model: {data['model']}")
        print(f"✅ Generated {len(data['answers'])} responses:\n")
        
        for i, answer in enumerate(data['answers'], 1):
            print(f"{'─' * 70}")
            print(f"Response {i}:")
            print(f"{answer}")
            print()
        
    except requests.exceptions.Timeout:
        print("⏰ Request timed out (model might be loading or generating)")
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
        if hasattr(e.response, 'text'):
            print(f"Error details: {e.response.text}")


def test_parameter_validation():
    """Test API parameter validation"""
    print_section("Testing Parameter Validation")
    
    # Test invalid num_responses
    print("\n[Test 1] Invalid num_responses (too many)")
    try:
        response = requests.post(
            f"{BASE_URL}/gpt2/answer/multiple",
            json={
                "question": "Test question?",
                "num_responses": 20,  # Should fail (max is 10)
            }
        )
        
        if response.status_code == 400:
            print("✅ Correctly rejected invalid num_responses")
            print(f"   Error: {response.json()['detail']}")
        else:
            print(f"⚠️  Expected 400 error, got {response.status_code}")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")
    
    # Test empty question
    print("\n[Test 2] Empty question")
    try:
        response = requests.post(
            f"{BASE_URL}/gpt2/answer",
            json={
                "question": "",
                "max_length": 150,
            }
        )
        
        print(f"Response status: {response.status_code}")
        if response.status_code == 200:
            print("⚠️  API accepted empty question (might want to add validation)")
        else:
            print("✅ Rejected empty question")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}")


def save_sample_output(filename: str = "gpt2_sample_output.json"):
    """Generate and save sample output for documentation"""
    print_section("Generating Sample Output for Documentation")
    
    sample_questions = [
        "What is deep learning?",
        "Who was Albert Einstein?",
        "How do neural networks work?",
    ]
    
    results = {
        "model": "gpt2-squad-finetuned",
        "examples": []
    }
    
    for question in sample_questions:
        print(f"📝 Processing: {question}")
        
        try:
            response = requests.post(
                f"{BASE_URL}/gpt2/answer",
                json={"question": question, "max_length": 150},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                results["examples"].append({
                    "question": question,
                    "answer": data["answer"]
                })
                print("   ✅ Success")
            else:
                print(f"   ❌ Failed with status {response.status_code}")
        
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Save to file
    with open(filename, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Sample output saved to: {filename}")


def main():
    """Run all tests"""
    print("\n" + "🚀" * 35)
    print("  Assignment 5: Fine-tuned GPT-2 API Testing")
    print("🚀" * 35)
    
    # Check if API is running
    if not test_api_health():
        print("\n❌ API is not running. Please start it with:")
        print("   uvicorn app.main:app --port 8000 --reload")
        return
    
    # Run tests
    test_gpt2_single_answer()
    test_gpt2_multiple_answers()
    test_parameter_validation()
    
    # Optional: Save sample output
    try:
        save_sample_output()
    except Exception as e:
        print(f"\n⚠️  Could not save sample output: {e}")
    
    # Summary
    print_section("Test Summary")
    print("✅ All tests completed!")
    print("\n📚 Next steps:")
    print("   1. Review the test results above")
    print("   2. Check gpt2_sample_output.json for sample responses")
    print("   3. Visit http://localhost:8000/docs for interactive API documentation")
    print("   4. Commit your code to GitHub")


if __name__ == "__main__":
    main()

