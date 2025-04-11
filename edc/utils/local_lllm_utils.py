from openai import OpenAI


def local_completion(text):
    client = OpenAI(
        # base_url="https://gpt-nha-lam-production.up.railway.app/v1",
        base_url="http://0.0.0.0:8080/v1",
        api_key="anything"
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": text}
        ]
    )

    return response.choices[0].message.contents

if __name__ == "__main__":
    print(local_completion(text="hello"))