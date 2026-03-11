import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/options.css";

function Options() {

  const navigate = useNavigate();

  return (

    <div className="options-container">

      <div className="options-card">

        <h1>Face Sketch System</h1>
        <p className="subtitle">
          Criminal Identification & Sketch Generation
        </p>

        <h2>Select an Option</h2>

        <div className="option-buttons">

          <div
            className="option-box"
            onClick={() => navigate("/create")}
          >
            <div className="icon">🎨</div>
            <h3>Create Sketch</h3>
            <p>Generate a facial sketch using face components</p>
          </div>

          <div className="option-box"
           onClick={()=>navigate("/upload")}
           >
            <div className="icon">📤</div>
            <h3>Upload Sketch</h3>
            <p>Upload an existing sketch for identification</p>
          </div>

        </div>

      </div>

    </div>

  );
}

export default Options;