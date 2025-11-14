import requests

requests.packages.urllib3.disable_warnings()
url = 'url'


api_key = 'api'

# 基于自测表 scl-90
# prompt = """You are a psychologist who needs to assess whether a patient's text contains depression related states and symptoms in order to score their depression status.Don't explain why, just output the result.
# symptoms:{headache, dizziness, poor memory, auditory hallucinations, insomnia, muscle soreness}
# state:{lack of interest in fun activities,negative self-talk or low self-esteem,eating more or less than usual,destructive impulse}
# response:0~10
# <example>
# text:I'd certainly kill myself.
# score:10.0

# text:every thing is so hard
# score:5.5
# </example>"""

prompt ="""You are a psychologist who needs to assess whether a patient's text contains depression related emotion in order to score based on the severity of emotions.Don't explain why, just output the result.
emotion:{sad,hopeless,failure,suicide,expected to be punished,guilty,cry,lose interest,worthless,low energy,change pattern,irritable,change appetite,tired}
response:0~5
<example>
text:Just It looks so lonely
score:3

text:I still feel awful whenever I make a mistake
score:5
</example>
"""



def send_message_r(message):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-2024-05-13",
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"{message}"}
        ]
    }
    response = requests.post(url, headers=headers, json=data, verify=False)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]['content']
    else:
        print(f"Error: {response.status_code}")
        return None

if __name__=="__main__":
    print("这测试要花钱的别瞎测")
    resp = send_message('I sometimes find that I am similar. If I get a good nights sleep then I seem to be punished for it the next day by being awake for about 24 hours even if I am tired after 18.  I am currently trying Magnesium Citrate and 5-HTP which are showing some early signs of being useful, have you tried those?')
    print(resp)
