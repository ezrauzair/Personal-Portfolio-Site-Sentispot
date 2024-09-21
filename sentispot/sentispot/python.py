from django.shortcuts import render, redirect
from django.http import JsonResponse
from django.contrib import messages
from django.contrib.messages import get_messages
import re
import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import base64
import io
import boto3
from transformers import BertTokenizer
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from googleapiclient.discovery import build

matplotlib.use('Agg')





def sentispot(request):
    return render(request, 'sentispot.html')


def homefakenews(request):
    return render(request, 'fakenews/fakenewshome.html')

def bulletchart(request):
    if request.method == 'POST' and 'process' in request.POST:
        # Clear existing messages
        storage = messages.get_messages(request)
        for _ in storage:
            pass
        access_key = 'Place Your Own AWS Access Key HERE'
        secret_key = 'Place Your Own AWS Secret Key HERE'
        region_name = 'Place Your Own AWS Region Name HERE'
        comprehend = boto3.client(
            'comprehend',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region_name
        )

        news = request.POST.get('news')

        if not news:
            messages.error(request, 'Please paste an article')
            return render(request, 'fakenews/fakenewshome.html')
        else:
            words = news.split()
            length = len(words)

        if length < 50:
            messages.error(request, 'The length is too short')
            return render(request, 'fakenews/fakenewshome.html')

        response = comprehend.detect_dominant_language(Text=news)
        detected_language_code = response['Languages'][0]['LanguageCode']
        confidence_score = response['Languages'][0]['Score']

        if detected_language_code != 'en' or confidence_score < 0.99:
            messages.error(request, 'An English article only')
            return render(request, 'fakenews/fakenewshome.html')
        

        device = torch.device('cpu')
        model = torch.load(r"D:\Porfolio Site\Full Website\sentispot\fakenewsmodelandtok\fakenewsdetectionmodel.pt", map_location=device)
        tokenizer = BertTokenizer.from_pretrained(r"sentispot/fakenewsmodelandtok/tok")

        model.eval().to('cpu')

        encoded_article = tokenizer.encode_plus(
            news,
            add_special_tokens=True,
            return_attention_mask=True,
            max_length=500,
            return_tensors='pt',
            padding='max_length',
            truncation=True
        )

        input_ids_real = encoded_article['input_ids']
        attention_masks_real = encoded_article['attention_mask']

        with torch.no_grad():
            input_ids_real = input_ids_real.to(device)
            attention_masks_real = attention_masks_real.to(device)

            outputs = model(input_ids_real, attention_mask=attention_masks_real)
            _, predicted_labels = torch.max(outputs.logits, 1)
            predictions = predicted_labels.cpu().numpy()
            probabilities = F.softmax(outputs.logits, dim=1).cpu().numpy()

        real_prob = probabilities[0][0]
        fake_prob = probabilities[0][1]

        labels = 'Real', 'Fake'
        sizes = [real_prob, fake_prob]
        colors = ['#0EB48B', 'lightcoral']
        explode = (0.1, 0)
        fig = plt.figure(figsize=(5, 4), dpi=100)
        plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.title('Real vs Fake Probability')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()

        image_base64 = base64.b64encode(image_png).decode('utf-8')
        chart = f'data:image/png;base64,{image_base64}'

        label = "Fake" if predictions[0] == 1 else "Real"
        
        
        if label == 'Real':
                return render(request, 'fakenews/fakenewshome.html', {'Real': label, 'realp': real_prob, 'fakep': fake_prob, 'chart': chart, 'ok':'ok'})
        elif label == "Fake":
                return render(request, 'fakenews/fakenewshome.html', {'Fake': label, 'realp': real_prob, 'fakep': fake_prob, 'chart': chart, 'ok':'ok'})
        else:
            return render(request, 'fakenews/fakenewshome.html')






















def homeyoutube(request):
    return render(request, 'youtubesentiment/youtubehome.html')


def videolinkandlangdetect(request):
    myapi = 'Place Your Own YouTube API Here'
    youtube = build('youtube', 'v3', developerKey=myapi)

    if "perform" in request.POST:
      
        link = request.POST.get('link')
        if not link:
            messages.success(request, 'Enter Video Link')  
            return render(request,'youtubesentiment/youtubehome.html')
    
        expression = r'(?:youtu\.be/|youtube\.com/watch\?v=)([A-Za-z0-9_-]+)'
        match = re.search(expression, link)

        if match:
            video_id = match.group(1)
        else:
            messages.success(request, 'Invalid Link') 
            return render(request,'youtubesentiment/youtubehome.html')
        
        # Retrieve all of the comments for the video
        next_page_token = None
        comments = []

        while True:
            response = youtube.commentThreads().list(
                part='snippet',
                videoId=video_id,
                textFormat='plainText',
                pageToken=next_page_token
            ).execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                comments.append(comment)

            next_page_token = response.get('nextPageToken')

            if not next_page_token:
                break

        # Store the comments in the session
        request.session['all_comments'] = comments
        
        english_comments = []
        access_key = 'Place Your Own AWS Access Key HERE'
        secret_key = 'Place Your Own AWS Secret Key HERE'
        region_name = 'Place Your Own AWS Region Name HERE'
        translate = boto3.client(
        'comprehend',
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name=region_name
        )
        
        for i in comments:
          if i.strip():
             response = translate.detect_dominant_language(Text=i)

             detected_language_code = response['Languages'][0]['LanguageCode']
             confidence_score = response['Languages'][0]['Score']

             if detected_language_code == 'en' and confidence_score > 0.99:
                 english_comments.append(i)       


        request.session['english_comments'] = english_comments
        if comments:
            
            return render(request, 'youtubesentiment/youtubehome.html', {'available': 'Your Comments are available','total_comments':len(english_comments)})
        else:
            return render(request, 'youtubesentiment/youtubehome.html', {'notavailable': 'No Comments for this video'}) 

    return render(request, 'youtubesentiment/youtubehome.html')




def viewcomments(request):

    comments = request.session.get('all_comments')
    
    if "view-comments" in request.POST:
            return render(request, 'youtubesentiment/allcommentspage.html', {'comments': comments})
    

    return render(request, 'youtubesentiment/youtubehome.html') 




def viewenglishcomments(request):

    english_comments = request.session.get('english_comments')
    
    if "english-comments" in request.POST:  
           if english_comments: 
                return render(request,'youtubesentiment/englishcommentspage.html',{'english_comments': english_comments}) 
           else: 
                return render(request,'youtubesentiment/youtubehome.html',{'english_comments_notavailable':'No English Comments'})  
  
    return render(request, 'youtubesentiment/youtubehome.html')




def piechart(request):
    english_comments = request.session.get('english_comments')
    df = pd.DataFrame(english_comments, columns=['text'])

    if 'piechartbutton' in request.POST:
        device = torch.device('cpu')
        mymodel = torch.load(r'C:\Users\PMYLS\Desktop\Full Website\sentispot\youtubemodelandtok\entiremodel.pt', map_location=device)
        tokenizer = BertTokenizer.from_pretrained(r"C:\Users\PMYLS\Desktop\Full Website\sentispot\youtubemodelandtok\token")
        mymodel.eval().to(device)

        # Tokenize input text
        encoded_inputs = tokenizer.batch_encode_plus(
        df['text'],
        add_special_tokens=True,
        return_attention_mask=True,
        max_length=200,
        return_tensors='pt',
        padding='max_length',
        truncation=True
         )

        # Extract tensors from encoded inputs
        attention_mask = encoded_inputs['attention_mask']
        input_ids = encoded_inputs['input_ids']
        dataset = TensorDataset(input_ids, attention_mask)
        loader = DataLoader(dataset, batch_size=20, shuffle=False)
        predictions = []
        predicted_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_masks = batch
                outputs = mymodel(input_ids, attention_mask=attention_masks)
                _, predicted = torch.max(outputs.logits, 1)
                predictions.extend(predicted.numpy())
        pos =[]
        neg = []
        neut = []

        for i in predictions:
          if i == 0:
             pos.append(i)
          elif i == 1:
             neg.append(i)
          elif i == 2:
             neut.append(i)


        # Process Predictions and Store Labels
        prediction_data = []
        predicted_labels = []

        for i, prediction in enumerate(predictions):
            sentiment_label = ["Positive", "Negative", "Neutral"][prediction] 
            predicted_labels.append(sentiment_label)
            prediction_data.append({'text': english_comments[i], 'sentiment': sentiment_label})

        request.session['prediction_data'] = prediction_data
        request.session['english_comments'] = english_comments   

        # Data for the donut chart
        values = [len(pos), len(neg), len(neut)]
        labels = ["Positive", "Negative", "Neutral"]
        colors = ['green', 'red', 'yellow']

        # Filter out categories with zero values
        non_zero_values = [val for val in values if val != 0]
        non_zero_labels = [label for label, val in zip(labels, values) if val != 0]

        # Create the donut chart
        plt.figure(figsize=(4.4, 4.4))
        plt.pie(non_zero_values, labels=non_zero_labels, colors=colors, autopct="%1.1f%%")

# Create a white circle in the center (the "donut hole")
        center_circle = plt.Circle((0, 0), 0.4, color='white')
        fig = plt.gcf()
        fig.gca().add_artist(center_circle)

# Generate Base64-encoded image data for the pie chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer
        piechart = base64.b64encode(buf.read()).decode('utf-8')

        # Data for the bar chart
        sentiment = ['Positive', 'Negative', 'Neutral', 'Total']
        count = [len(pos), len(neg), len(neut), len(english_comments)]
        # Create a bar chart
        bar_width = 0.4  # Adjust this value as needed
        plt.figure(figsize=(4, 4))
        plt.bar(sentiment, count, color=['green', 'red', 'yellow', 'blue'], width=bar_width)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)

        # Generate Base64-encoded image data for the bar chart
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)  # Rewind the buffer
        barchart = base64.b64encode(buf.read()).decode('utf-8')

        return render(request, 'youtubesentiment/youtubehome.html', {
                'pos': len(pos), 'neg': len(neg), 'neut': len(neut),
                'total_comments': len(english_comments),
                'piechart': piechart, 'barchart': barchart,
            })
    else:
        return render(request, 'youtubesentiment/youtubehome.html',{'ok':'okmessage'})  # Or return any response you want if the button is not pressed



def predictions(request):
    if request.method == 'POST':
        english_comments = request.session.get('english_comments', [])
        prediction_data = request.session.get('prediction_data', [])
        return render(request, 'youtubesentiment/predictions.html', {'english_comments': english_comments, 'predictions': prediction_data})
    else:
        return redirect('home')



def signup(request):
    if request.method == 'POST':
        return render(request, 'youtubesentiment/youtubehome.html', {'show_container1': True})
    else:
        return render(request, 'youtubesentiment/youtubehome.html', {'show_container1': False})


def login(request):
    if request.method == 'POST':
        return render(request, 'youtubesentiment/youtubehome.html', {'show_container2': True})
    else:
        return render(request, 'youtubesentiment/youtubehome.html', {'show_container2': False})    













