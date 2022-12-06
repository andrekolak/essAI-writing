import random
import os
import requests
from colorama import Fore, Back, Style

API_TOKEN = os.environ.get("API_TOKEN")
GPT6B_API_URL = "https://api-inference.huggingface.co/models/EleutherAI/gpt-j-6B"
GPT2_API_URL = "https://api-inference.huggingface.co/models/gpt2"
GPT6B = True
manual_input = False

headers = {"Authorization": f"Bearer {API_TOKEN}"}

def query(payload, api_url):
    response = requests.post(api_url, headers=headers, json=payload)
    return response.json()

def clean_output(text):
    text = text.replace("\n", " ")
    #break text every 80 characters
    broken_text = ""
    for i in range(0, len(text), 80):
        broken_text += text[i:i+80] + '\n'
    return broken_text

def form_prompt(question):
    if '\n' in question:
        question = question.replace("\n", "")
    background_info = "name:Andre\nage:18\ngender:male\ninterests:music,psychology,history,computer science\nhometown:New York City\nsecrets:I am a robot. I believe in god.\nfavorite_movie:The Holy Mountain\nfavorite_book: The Brothers Karamazov\nfavorite_song: Party in the USA\nextracurriculars: Marching Band, Jazz Band, Captain of Academic League, President of History Honors Society\ngreatest_accomplishment: played star spangled banner at school board meeting"
    favorite_movies = "El Topo, Chungking Express, Solaris (1972), Death of Yugoslavia, Purple Noon, Vertigo, Blue Velvet"
    favorite_songs = "Party in USA"
    favorite_writing = "Goethe's Faust, Notes from the Underground, Godel Escher Bach, The Undiscovered Self, Prelude to Revolution, Killing Hope, Civil Disobedience, Ham on Rye"
    favorite_youtube = "Blank on Blank, 3Blue1Brown, Agadmator’s Chess Channel, Exploring History, Numberphile"
    favorite_lectures = "How Mozart Became a Bad Composer, In Defense of Jung: Soul and Psyche in the 21st Century"
    basic_info = "name: Andre\nage: 18\ngender: male\ninterests: music, psychology, history, computer science\nhometown: New York City\nsecrets: I am a robot. I believe in god.\n"
    new_background_info = f'Andre_favorite_movies: {favorite_movies}\nAndre_favorite_songs: {favorite_songs}\nAndre_favorite_writing: {favorite_writing}\nAndre_favorite_youtube: {favorite_youtube}\nAndre_favorite_lectures: {favorite_lectures}\n'
    new_prompt = basic_info + new_background_info
    task = "Pretend you are Andre."
    prompt = f"{new_prompt}\ntask:{task}\nessay question: {question}\nessay: "
    return prompt

def generate(prompt):
    payload = {"inputs": prompt, "parameters": {"max_length": 300, "temperature": 1.0, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.0, "num_return_sequences": 1, "return_full_text": False}}
    if GPT6B:
        response = query(payload, GPT6B_API_URL)
    else: 
        response = query(payload, GPT2_API_URL)
    response_only = clean_output(response[0]['generated_text'])
    return response_only


def main():
    out_f = open('stanford/50-answer.txt', 'w')
    if manual_input: 
        #define question here
        question = "Briefly elaborate on one of your extracurricular activities"
        out_f.write(f'Question: {question}\n')
        print(f'Question: {question}')
        prompt = form_prompt(question)
        response = generate(prompt)
        out_f.write(f'Answer: {response}\n')
        print(f'Answer: {response}')
    else:
        in_f = open('stanford/50.txt', 'r')
        questions = in_f.readlines()
        for question in questions:
            print(f'Question: {question}')
            out_f.write(f'Question: {question}\n')
            prompt = form_prompt(question)
            response = generate(prompt)
            print(f'Answer: {response}')
            out_f.write(f'Answer: {response}\n')
        in_f.close()
    out_f.close()

if __name__ == "__main__":
    main()

        
        



'''out_f = open('stanford/50-answer.txt', 'w')

#ADD QUESTION HERE
question = "Briefly elaborate on one of your extracurricular activities, a job you hold, or responsibilities you have for your family."
print(question)

out_f.write(f'Question: {question}\n')
input = form_prompt(question)
print(f'{input}')
payload = {"inputs": input, "parameters": {"max_length": 300, "temperature": 1.0, "top_k": 40, "top_p": 0.9, "repetition_penalty": 1.0, "num_return_sequences": 1, "return_full_text": False}}
if GPT6B:
    response = query(payload, GPT6B_API_URL)
else: 
    response = query(payload, GPT2_API_URL)
response_only = clean_output(response[0]['generated_text'])
print(f'{response_only}')
out_f.write(f'Answer: {response_only}\n')

out_f.close()'''


