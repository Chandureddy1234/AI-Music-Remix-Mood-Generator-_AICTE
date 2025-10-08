# 🎵 AI Music Remix & Mood Generator

Create, remix, and transform music using **100% FREE AI models**. No technical skills required - perfect for students!

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)
![Status](https://img.shields.io/badge/status-production%20ready-brightgreen.svg)

---

## 🎯 Overview

**The Problem:**
Students need expensive software and technical skills to create or remix music.

**The Solution:**
A simple web app where anyone can:
- 🎵 Generate music from text ("upbeat dance music with synths")
- 🎚️ Remix songs with AI (separate vocals, change genre, add effects)
- 🎯 Analyze mood and get creative suggestions
- 🎨 Layer and mix multiple tracks

**100% FREE** • **No Installation** • **No Music Skills Needed**

---

## ✨ Key Features

### 🎼 Music Generation
- Generate music from text descriptions
- 15+ genres, 15+ moods
- Control BPM, duration (15-120s), and musical key
- Batch generation for variations
- Uses Meta's MusicGen (local or cloud)

### 🎚️ Remix Engine  
- Upload audio files (MP3, WAV, FLAC, OGG)
- AI stem separation (vocals, drums, bass, instruments) with Demucs
- Genre transfer (pop → lo-fi, rock → jazz)
- Mood transformation
- Tempo/pitch control
- Audio effects (reverb, echo, distortion, lo-fi filter)

### 🎯 Mood Analyzer
- Auto-detect mood and genre
- Extract BPM, key, energy, danceability
- Waveform and spectrogram visualizations
- Get AI remix suggestions

### 🎨 Creative Studio
- Layer multiple tracks
- Real-time mixing
- Volume controls per track
- Export in multiple formats

### � History & More
- Save favorites
- Search and filter generations
- Mobile-responsive UI
- Dark mode interface

---

## 🆓 100% Free Technology

| Component | Provider | Cost |
|-----------|----------|------|
| Music Generation | Meta MusicGen | FREE |
| Stem Separation | Demucs | FREE |
| Cloud APIs | HuggingFace, Replicate | FREE tiers |
| LLM Enhancement | Groq (Llama 3.1) | FREE 30 req/min |
| Audio Effects | Pedalboard (Spotify) | FREE |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- 4GB RAM (8GB recommended)
- Internet connection

### Installation (2 minutes)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd "AI music generator"

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
```

Opens at `http://localhost:8501`

### Optional: Add API Keys (All FREE)

Get a Groq API key for better prompts (30 req/min free):
1. Visit https://console.groq.com
2. Sign up and create API key
3. Create `.env` file: `GROQ_API_KEY=gsk_your_key`

**Other free options:**
- HuggingFace: https://huggingface.co/settings/tokens
- Replicate: https://replicate.com/account/api-tokens ($10 free credits)

---

## 📖 How to Use

### Generate Music
1. Go to **"Music Generation"** page
2. Type: "chill lo-fi hip hop beat"
3. Select genre and mood
4. Click **"Generate"**
5. Download your track!

### Remix a Song
1. Go to **"Remix Engine"** page
2. Upload audio file
3. Choose: Separate stems / Change genre / Add effects
4. Download remix

### Analyze Music
1. Go to **"Mood Analyzer"** page  
2. Upload audio
3. View mood, BPM, energy, and visualizations

---

## 📁 Project Structure

```
AI music generator/
├── app.py                 # Main application
├── music_generator.py     # AI music generation
├── audio_processor.py     # Audio processing & effects
├── components.py          # UI components
├── config.py              # Configuration
├── requirements.txt       # Dependencies
└── README.md             # This file
```

---

## 🎓 Perfect for Students

✅ **No music theory needed** - Just describe what you want  
✅ **No expensive software** - Everything is free  
✅ **Quick results** - Generate in 30-60 seconds  
✅ **Learn by doing** - Experiment with genres and moods  
✅ **School projects** - Create music for presentations  

---

## 🔧 Troubleshooting

**Models downloading slowly?**  
First run downloads models (~500MB), then cached locally.

**Out of memory?**  
Use cloud mode or smaller model size in settings.

**API key error?**  
Check `.env` file or add keys in Settings page.

---

## 🚀 Deploy to Streamlit Cloud (FREE Hosting)

```bash
# 1. Push to GitHub
git init
git add .
git commit -m "AI Music Generator"
git push origin main

# 2. Deploy
# Visit streamlit.io/cloud
# Connect GitHub repo
# Add API keys in Secrets
# Deploy!
```

Your app gets a free URL: `https://your-app.streamlit.app`

---

## 🤝 Contributing

Pull requests welcome! Please:
1. Fork the repo
2. Create feature branch
3. Test changes
4. Submit PR

---

## 📄 License

MIT License - Free to use, modify, and distribute!

---

## 🙏 Credits

- **Meta AI** - MusicGen & Demucs
- **Groq** - Fast LLM inference  
- **Spotify** - Pedalboard audio effects
- **Streamlit** - Web framework
- **HuggingFace** - Model hosting

---

## 📊 Stats

- **8,000+ lines of code**
- **25+ features**
- **15+ genres & moods**
- **100% FREE to use**

---

**Made with ❤️ for students and music lovers**

**⭐ Star this repo if you find it helpful!**
