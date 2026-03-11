import React, { useState } from "react";
import axios from "axios";
import "../styles/uploadSketch.css";

function UploadSketch() {

  const [file, setFile] = useState(null);
  const [matches, setMatches] = useState([]);
  const [preview, setPreview] = useState(null);

  const handleFileChange = (e) => {

    const selected = e.target.files[0];

    setFile(selected);

    if (selected) {
      setPreview(URL.createObjectURL(selected));
    }

  };

  const uploadSketch = async () => {

    if (!file) {
      alert("Please select a sketch image");
      return;
    }

    const formData = new FormData();
    formData.append("sketch", file);

    try {

      const res = await axios.post(
        "http://localhost:5000/api/identify",
        formData
      );

      setMatches(res.data.matches || []);

    } catch (err) {

      console.error("Upload error:", err);

    }

  };

  const topMatch = matches.length > 0 ? matches[0] : null;

  return (

    <div className="container">

      <h1 className="title">Face Sketch Identification</h1>

      <div className="uploadCard">

        <input
          type="file"
          className="fileInput"
          onChange={handleFileChange}
        />

        <button className="button" onClick={uploadSketch}>
          Identify Face
        </button>

      </div>

      {preview && (

        <div className="resultContainer">

          {/* Uploaded Sketch */}

          <div className="resultBox">

            <h3>Uploaded Sketch</h3>

            <img
              src={preview}
              className="imageLarge"
              alt="sketch"
            />

          </div>

          {/* Top Match */}

          {topMatch && (

            <div className="resultBox">

              <h3>Top Match</h3>

              <img
                src={`http://localhost:5000/photos/${topMatch.name}`}
                className="imageLarge"
                alt="match"
              />

              <p className="score">
                Similarity: {(topMatch.score * 100).toFixed(2)}%
              </p>

            </div>

          )}

        </div>

      )}

    </div>

  );
}

export default UploadSketch;