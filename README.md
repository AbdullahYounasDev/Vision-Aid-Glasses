<div style="background-color:#f5f5f5; padding:20px; border-radius:10px;">

<h2 style="color:#1f4e79;">YOLOv5 Minimal Detection with eSpeak NG</h2>

<p>This project uses YOLOv5 for object detection and eSpeak NG for voice notifications. Follow the guide below to set it up on Windows.</p>

</div>

---

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">1️⃣ Clone or Pull the Repository</h3>

<p><strong>Clone for the first time:</strong></p>
<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>git clone https://github.com/your-username/your-repo-name.git</code></pre>

<p><strong>Pull latest changes if you already have the repo:</strong></p>
<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>cd your-repo-name
git pull origin main</code></pre>

</div>

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">2️⃣ Change Directory</h3>

<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>cd your-repo-name</code></pre>

</div>

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">3️⃣ Install Python Dependencies</h3>

<p>Make sure Python (>=3.10) is installed. Then run:</p>
<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>pip install -r requirements.txt</code></pre>

</div>

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">4️⃣ Install eSpeak NG (Windows)</h3>

<p>Install via <strong>winget</strong>:</p>
<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>Win + R → cmd → Enter
winget install espeak-ng</code></pre>

<p>eSpeak NG will be installed to <code>C:\Program Files\eSpeak NG</code></p>

</div>

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">5️⃣ Optional: Set Environment Variable</h3>

<p>To set up eSpeak NG on Windows, press <strong>Win + R</strong>, type <code>sysdm.cpl</code> and hit Enter to open System Properties. Go to the <strong>Advanced</strong> tab and click on <strong>Environment Variables</strong>. Under <strong>System variables</strong>, click <strong>New</strong> and add <code>C:\Program Files\eSpeak NG\</code> as the value. Click OK to save and apply the changes. This ensures Python can find eSpeak NG when running the detection script.</p>


</div>

<div style="background-color:#eef6fb; padding:15px; border-left:5px solid #1f4e79; margin-bottom:10px;">

<h3 style="color:#1f4e79;">6️⃣ Run Detection</h3>

<pre style="background-color:#dbe9f4; padding:10px; border-radius:5px;"><code>python detect-minimal.py --weights yolov5m.pt --source 0</code></pre>

<ul>
<li><code>--weights</code> → choose your model (yolov5s.pt, yolov5m.pt, etc.)</li>
<li><code>--source</code> → <code>0</code> for webcam, or path to image/video</li>
</ul>

<p>Press <strong>q</strong> to close the OpenCV window.</p>

</div>

<div style="background-color:#f5f5f5; padding:20px; border-radius:10px; margin-top:10px;">

<p>✅ You are now ready! Adjust voice speed or language inside <code>detect-minimal.py</code> as needed.</p>

</div>
