import React from "react";
import TopButtons from "./TopButtons";
import ImageGrid from "./ImageGrid";

function RightPanel({
  selectedPart,
  setSelectedPart,
  elements,
  setElements,
  history,
  setHistory
}) {

  const deleteShape = () => {

    if (!selectedPart) {
      alert("Select a shape first");
      return;
    }

    // save state for undo
    setHistory(prev => [...prev, elements]);

    setElements(prev => {
      const updated = { ...prev };
      delete updated[selectedPart];
      return updated;
    });

    setSelectedPart(null);
  };

  return (
    <div className="right-panel">

      <TopButtons
        setElements={setElements}
        history={history}
        setHistory={setHistory}
      />

      <div className="section-header">

        <h4>
          {selectedPart ? selectedPart.toUpperCase() : "FACE"} SHAPES TO CHOOSE FROM
        </h4>

        <button
          className="delete-btn"
          onClick={deleteShape}
        >
          Delete Shape
        </button>

      </div>

      <ImageGrid part={selectedPart} />

    </div>
  );
}

export default RightPanel;