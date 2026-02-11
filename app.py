import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pickle
import torch.nn.functional as F

# --- 1. VOCABULARY CLASS ---
class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold
    def __len__(self): return len(self.itos)

# --- 2. MODEL ARCHITECTURE ---
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
    def forward(self, images): return self.bn(self.linear(images))

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

# --- 3. LOAD RESOURCES ---
@st.cache_resource
def load_resources():
    with open('vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNtoRNN(512, 512, len(vocab), 1).to(device)
    model.load_state_dict(torch.load('flickr30k_model.pth', map_location=device))
    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1]).to(device)
    resnet.eval()
    return model, resnet, vocab, device, transform

model, resnet, vocab, device, transform = load_resources()

# --- 4. SEARCH ALGORITHMS ---
def greedy_search(image):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image).view(1, -1)
        x = model.encoderCNN(features).unsqueeze(1)
        states = None
        result_caption = []
        for _ in range(20):
            hiddens, states = model.decoderRNN.lstm(x, states)
            output = model.decoderRNN.linear(hiddens.squeeze(1))
            predicted = output.argmax(1)
            word = vocab.itos[predicted.item()]
            if word == "<EOS>": break
            if word not in ["<SOS>", "<PAD>"]: result_caption.append(word)
            x = model.decoderRNN.embed(predicted).unsqueeze(1)
    return " ".join(result_caption)

def beam_search(image, beam_width=5):
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(image).view(1, -1)
        encoded_features = model.encoderCNN(features).unsqueeze(1)
        beams = [([vocab.stoi["<SOS>"]], 0, None)]
        for _ in range(20):
            new_beams = []
            for seq, score, states in beams:
                if seq[-1] == vocab.stoi["<EOS>"]:
                    new_beams.append((seq, score, states))
                    continue
                last_word = torch.tensor([seq[-1]]).to(device)
                x = encoded_features if len(seq) == 1 else model.decoderRNN.embed(last_word).unsqueeze(1)
                hiddens, states = model.decoderRNN.lstm(x, states)
                outputs = model.decoderRNN.linear(hiddens.squeeze(1))
                log_probs = F.log_softmax(outputs, dim=1)
                top_v, top_i = log_probs.topk(beam_width)
                for i in range(beam_width):
                    new_beams.append((seq + [top_i[0][i].item()], score + top_v[0][i].item(), states))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
            if all(b[0][-1] == vocab.stoi["<EOS>"] for b in beams): break
    best_seq = beams[0][0]
    return " ".join([vocab.itos[i] for i in best_seq if i not in [vocab.stoi["<SOS>"], vocab.stoi["<EOS>"], vocab.stoi["<PAD>"]]])

# --- 5. INITIALIZE SESSION STATE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- 6. ADVANCED VIBRANT UI ---
st.set_page_config(page_title="Neural Storyteller", page_icon="🎨", layout="centered")

st.markdown("""
    <style>
    /* Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #FDFCFB 0%, #E2D1F9 100%);
    }
    
    /* Main Title Style */
    h1 {
        color: #6A5ACD;
        font-family: 'Playfair Display', serif;
        font-size: 3rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 0px;
    }
    
    /* Caption Card Styling */
    .caption-box {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        padding: 25px;
        border: 2px solid #D8BFD8;
        box-shadow: 0 10px 30px rgba(106, 90, 205, 0.1);
        margin-top: 20px;
        transition: transform 0.3s ease;
    }
    .caption-box:hover { transform: translateY(-5px); }
    
    /* Custom Button Styling */
    div.stButton > button:first-child {
        background: linear-gradient(45deg, #FF9A9E 0%, #FAD0C4 99%, #FAD0C4 100%);
        color: white;
        border-radius: 30px;
        border: none;
        font-weight: bold;
        padding: 0.6rem 2rem;
        font-size: 1.2rem;
        width: 100%;
        box-shadow: 0 4px 15px rgba(255, 154, 158, 0.4);
    }
    
    /* Clear Button Styling */
    .clear-btn > button {
        background: #E6E6FA !important;
        color: #6A5ACD !important;
        font-size: 0.8rem !important;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🎨 Neural Storyteller")
st.markdown("<p style='text-align: center; color: #9370DB;'>Where Computer Vision meets Poetry</p>", unsafe_allow_html=True)

# --- 7. SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")
    method = st.radio("Captioning Strategy", ["Greedy (Instant)", "Beam (High Quality)"])
    st.divider()
    if st.button("🧹 Clear Gallery", key="clear_all"):
        st.session_state.history = []
        st.rerun()

# --- 8. UPLOAD & GENERATE ---
uploaded_file = st.file_uploader("Upload an image to start the story...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Use a column layout for the current upload
    img = Image.open(uploaded_file).convert("RGB")
    
    container = st.container()
    with container:
        st.image(img, use_container_width=True)
        
        if st.button("✨ Generate Story"):
            with st.spinner("Analyzing pixels and drafting words..."):
                if "Greedy" in method:
                    caption = greedy_search(img)
                else:
                    caption = beam_search(img)
                
                # Append to history (newest first)
                st.session_state.history.insert(0, {
                    "image": img,
                    "caption": caption,
                    "method": method
                })
                # Note: No rerun here so we see the "Generate" effect immediately, 
                # but history is updated for next render.

# --- 9. THE MEMORY GALLERY ---
if st.session_state.history:
    st.markdown("## 🌸 Memory Gallery")
    for idx, item in enumerate(st.session_state.history):
        # Create a card for each item
        cols = st.columns([1, 1.5])
        with cols[0]:
            st.image(item["image"], use_container_width=True)
        with cols[1]:
            st.markdown(f"""
            <div class="caption-box">
                <span style="color: #FFB6C1; font-weight: bold; font-size: 0.8rem;">{item['method'].upper()}</span>
                <p style="font-size: 1.4rem; color: #4B0082; margin-top: 10px;">"{item['caption']}"</p>
            </div>
            """, unsafe_allow_html=True)
        st.divider()
else:
    st.info("The gallery is empty. Upload your first image above! ☝️")