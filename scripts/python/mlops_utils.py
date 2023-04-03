import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_gpt_code_from_prompt(prompt, wrapper, model="gpt-3.5-turbo"):
  completion = openai.ChatCompletion.create(
    model=model,
    messages=[
      {"role": "user", "content": f"{wrapper} {prompt}"}
    ]
  )
  return completion.choices[0].message.content