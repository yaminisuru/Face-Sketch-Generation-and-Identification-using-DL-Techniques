import React from "react";
import html2canvas from "html2canvas";


function TopButtons({ setElements, history, setHistory }) {

  // ✅ Reset should clear object
  const reset = () => {
    setElements({});
  };

  // ✅ Save canvas image
  const saveImage = () => {
    const canvas = document.getElementById("canvas-box");

    if (!canvas) return;

    html2canvas(canvas).then((canvasImg) => {
      const link = document.createElement("a");
      link.download = "sketch.png";
      link.href = canvasImg.toDataURL("image/png");
      link.click();
    });
  };

  return (
    <div className="top-buttons">
      <button onClick={saveImage}>SAVE</button>
      <button onClick={reset} style={{ marginLeft: "10px" }}>
        RESET
      </button>
      <button>Compare</button>
    </div>
  );
}

export default TopButtons;