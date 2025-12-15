from flask import Flask, render_template, request, jsonify, send_from_directory
import torch
import os
import json
from werkzeug.utils import secure_filename
from utils import preprocess_video, DeepfakeDetector
import traceback
import mimetypes

# Ensure MP4 MIME type is registered
mimetypes.add_type('video/mp4', '.mp4')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching for development

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global model variable
model = None
device = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    """Load the model for web app"""
    global model, device
    
    # Use the checkpoint that matches our model.py architecture
    MODEL_PATH = "model.pt"
    
    print(f"Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = DeepfakeDetector(num_classes=2)
    
    # Load the checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    # Load state dict directly (keys should match now)
    model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    model.eval()
    print("Model loaded successfully!")
    
    return model

def predict_video(video_path, sequence_length=60):
    """Make prediction on a video file"""
    # Preprocess video (extract frames)
    video_tensor = preprocess_video(video_path, sequence_length=sequence_length, im_size=112)
    video_tensor = video_tensor.to(device)
    
    # Make prediction
    with torch.no_grad():
        output = model(video_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred = torch.argmax(probabilities, dim=1).item()
        
        result = {
            'prediction': 'REAL' if pred == 1 else 'FAKE'
        }
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['video']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: mp4, avi, mov, mkv, webm'}), 400
        
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process the video
        print(f"Processing video: {filepath}")
        
        sequence_length = int(request.form.get('sequence_length', 60))
        result = predict_video(filepath, sequence_length)
        result['filename'] = filename
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error processing video: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500

@app.route('/test-dataset')
def test_dataset():
    """Page for testing on our dataset samples"""
    # Load sample videos metadata
    with open('sample_videos.json', 'r') as f:
        sample_videos = json.load(f)
    return render_template('test_dataset.html', videos=sample_videos)

@app.route('/test-sample/<int:video_id>', methods=['POST'])
def test_sample(video_id):
    """Test a sample video from our dataset"""
    try:
        # Load sample videos metadata
        with open('sample_videos.json', 'r') as f:
            sample_videos = json.load(f)
        
        # Find the video
        video = next((v for v in sample_videos if v['id'] == video_id), None)
        if not video:
            return jsonify({'error': 'Video not found'}), 404
        
        # Get video path
        video_path = os.path.join('static', 'dataset_videos', video['filename'])
        
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        print(f"Testing sample video: {video['filename']}")
        
        # Make prediction
        sequence_length = int(request.form.get('sequence_length', 30))
        result = predict_video(video_path, sequence_length)
        
        # Add video info
        result['filename'] = video['title']
        result['true_label'] = video['true_label']
        result['dataset'] = video['dataset']
        result['description'] = video['description']
        
        # Check if prediction is correct
        result['is_correct'] = result['prediction'] == video['true_label']
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error testing sample: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': f'Error testing sample: {str(e)}'}), 500

@app.route('/get-sample-videos')
def get_sample_videos():
    """API endpoint to get list of sample videos"""
    try:
        with open('sample_videos.json', 'r') as f:
            sample_videos = json.load(f)
        return jsonify(sample_videos)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/model-info')
def model_info():
    """Endpoint to show which model is being used"""
    info = {
        'status': 'Model loaded successfully',
        'purpose': 'Deepfake detection'
    }
    return jsonify(info)

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    print("=" * 80)
    print("🚀 Deepfake Detection Web Application")
    print("=" * 80)
    
    # Load the model on startup
    load_model()
    
    print("\n✅ Server ready!")
    print("   • Upload your own videos: http://localhost:5002")
    print("   • Test on our dataset: http://localhost:5002/test-dataset")
    print("=" * 80)
    app.run(debug=True, host='0.0.0.0', port=5001)
