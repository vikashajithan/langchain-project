import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
import re

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="YouTube Summarizer",
    page_icon="📺",
    layout="centered"
)

# Title and description
st.title("📺 YouTube Video Summarizer")
st.caption("Powered by Groq + LangChain")
st.markdown("---")

def get_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com/watch\?v=|youtu\.be/|youtube\.com/embed/)([^&\n?#]+)',
        r'youtube\.com/watch\?.*v=([^&\n?#]+)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_transcript(video_id):
    """Fetch transcript using multiple fallback methods"""
    
    # Method 1: Try youtube-transcript-api
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
        
        # Get all available transcripts
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        
        # Try manual transcripts first (more accurate)
        try:
            transcript = transcript_list.find_manually_created_transcript(['en', 'en-US', 'en-GB'])
            transcript_data = transcript.fetch()
        except:
            # Fall back to auto-generated
            try:
                transcript = transcript_list.find_generated_transcript(['en', 'en-US', 'en-GB'])
                transcript_data = transcript.fetch()
            except:
                # Get first available transcript
                for t in transcript_list:
                    transcript_data = t.fetch()
                    break
        
        transcript_text = " ".join([item["text"] for item in transcript_data])
        return transcript_text
        
    except Exception as e1:
        st.warning(f"Method 1 failed: {str(e1)}")
        
        # Method 2: Try with requests and BeautifulSoup
        try:
            import requests
            from bs4 import BeautifulSoup
            import json
            
            url = f"https://www.youtube.com/watch?v={video_id}"
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract captions from page data
            scripts = soup.find_all('script')
            for script in scripts:
                if 'captionTracks' in str(script):
                    # Parse and extract caption URL
                    # This is a simplified approach
                    pass
            
            raise Exception("BeautifulSoup method not fully implemented")
            
        except Exception as e2:
            st.warning(f"Method 2 failed: {str(e2)}")
            
            # Method 3: Try yt-dlp (most reliable)
            try:
                import yt_dlp
                
                ydl_opts = {
                    'skip_download': True,
                    'writesubtitles': True,
                    'writeautomaticsub': True,
                    'subtitleslangs': ['en'],
                    'quiet': True,
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
                    
                    # Get subtitles
                    if 'subtitles' in info and 'en' in info['subtitles']:
                        subtitle_url = info['subtitles']['en'][0]['url']
                        import requests
                        response = requests.get(subtitle_url)
                        
                        # Parse VTT or SRT format
                        transcript_text = response.text
                        # Clean up timestamps and formatting
                        lines = transcript_text.split('\n')
                        text_lines = [line for line in lines if not line.startswith('WEBVTT') 
                                     and not '-->' in line and line.strip() and not line.strip().isdigit()]
                        transcript_text = ' '.join(text_lines)
                        return transcript_text
                    
                    # Try automatic captions
                    if 'automatic_captions' in info and 'en' in info['automatic_captions']:
                        subtitle_url = info['automatic_captions']['en'][0]['url']
                        import requests
                        response = requests.get(subtitle_url)
                        
                        transcript_text = response.text
                        lines = transcript_text.split('\n')
                        text_lines = [line for line in lines if not line.startswith('WEBVTT') 
                                     and not '-->' in line and line.strip() and not line.strip().isdigit()]
                        transcript_text = ' '.join(text_lines)
                        return transcript_text
                
                raise Exception("No subtitles found")
                
            except Exception as e3:
                raise Exception(f"All methods failed. Last error: {str(e3)}")

def create_summary_chain():
    """Create LangChain summarization chain"""
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("Groq API key not found in .env file")
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        groq_api_key=groq_api_key,
        max_tokens=1024
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at summarizing YouTube video content. 
        Create a clear, well-structured summary that captures the key points and main ideas.
        Format your summary with:
        - A brief overview (2-3 sentences)
        - Key points (bullet points)
        - Main takeaways or conclusion
        """),
        ("human", "Summarize the following YouTube video transcript:\n\n{transcript}")
    ])
    
    chain = prompt | llm | StrOutputParser()
    
    return chain

# Main interface
youtube_url = st.text_input(
    "🔗 Enter YouTube Video URL:",
    placeholder="https://www.youtube.com/watch?v=..."
)

# Display video preview if URL is provided
if youtube_url:
    video_id = get_video_id(youtube_url)
    if video_id:
        st.video(youtube_url)
    else:
        st.warning("⚠️ Invalid YouTube URL format")

# Summary button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    summarize_button = st.button("🎯 Generate Summary", use_container_width=True)

# Process summarization
if summarize_button:
    if not youtube_url:
        st.warning("⚠️ Please enter a YouTube URL")
    else:
        video_id = get_video_id(youtube_url)
        
        if not video_id:
            st.error("❌ Invalid YouTube URL. Please check and try again.")
        else:
            try:
                # Step 1: Fetch transcript
                with st.status("Processing video...", expanded=True) as status:
                    st.write("📥 Fetching transcript...")
                    transcript = get_transcript(video_id)
                    
                    st.write(f"✅ Transcript retrieved ({len(transcript)} characters)")
                    
                    # Step 2: Create chain and generate summary
                    st.write("🤖 Generating summary with AI...")
                    chain = create_summary_chain()
                    summary = chain.invoke({"transcript": transcript})
                    
                    status.update(label="✨ Summary generated!", state="complete")
                
                # Display summary
                st.markdown("---")
                st.subheader("📄 Summary")
                st.markdown(summary)
                
                # Optional: Show transcript
                with st.expander("📜 View Full Transcript"):
                    st.text_area(
                        "Transcript",
                        transcript,
                        height=300,
                        disabled=True
                    )
                
            except ValueError as ve:
                st.error(f"❌ Configuration Error: {str(ve)}")
                st.info("💡 Please make sure GROQ_API_KEY is set in your .env file")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.info("💡 Make sure the video has captions/subtitles available")

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>Built with Streamlit, LangChain & Groq</div>",
    unsafe_allow_html=True
)