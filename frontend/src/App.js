import React, { useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [videoURL, setVideoURL] = useState("");
  const [loading, setLoading] = useState(false);

  const handleFileChange = (e) => {
    const selected = e.target.files[0];
    setFile(selected);
    setVideoURL(URL.createObjectURL(selected)); // preview video
    setPrediction(""); // reset previous prediction
  };

  const handleUpload = async () => {
    if (!file) return alert("Please select a video first!");
    setLoading(true);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.prediction) {
        setPrediction(data.prediction);
      } else if (data.error) {
        alert(data.error);
      }
    } catch (err) {
      console.error(err);
      alert("Error predicting video. Make sure backend is running!");
    }
    setLoading(false);
  };

  return (
    <div
      style={{
        fontFamily: "Arial, sans-serif",
        textAlign: "center",
        marginTop: "50px",
        backgroundColor: "#f5f5f5",
        minHeight: "100vh",
        padding: "50px",
      }}
    >
      <h1 style={{ color: "#333", marginBottom: "30px" }}>
        🎬 Action Recognition Demo
      </h1>

      <div style={{ marginBottom: "20px" }}>
        <input
          type="file"
          accept="video/*"
          onChange={handleFileChange}
          style={{ padding: "10px", borderRadius: "5px" }}
        />
      </div>

      {videoURL && (
        <div style={{ marginBottom: "20px" }}>
          <video
            width="500"
            controls
            src={videoURL}
            style={{ borderRadius: "10px", boxShadow: "0px 0px 10px #aaa" }}
          >
            Your browser does not support the video tag.
          </video>
        </div>
      )}

      <button
        onClick={handleUpload}
        style={{
          padding: "12px 25px",
          fontSize: "16px",
          backgroundColor: "#4CAF50",
          color: "white",
          border: "none",
          borderRadius: "8px",
          cursor: "pointer",
          boxShadow: "0px 4px 6px #888",
        }}
        disabled={loading}
      >
        {loading ? "Predicting..." : "Predict Action"}
      </button>

      {prediction && (
        <h2
          style={{
            marginTop: "30px",
            color: "#ff5722",
            fontSize: "24px",
            fontWeight: "bold",
          }}
        >
          ✅ Prediction: {prediction}
        </h2>
      )}
    </div>
  );
}

export default App;
