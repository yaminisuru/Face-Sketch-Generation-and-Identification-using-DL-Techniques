import React from "react";
import { useDrop } from "react-dnd";
import { Rnd } from "react-rnd";

function CanvasArea({
  elements,
  setElements,
  history,
  setHistory,
  selectedPart,
  setSelectedPart
}) {

  const defaultPositions = {
    head: { x: 150, y: 50, width: 200 },
    hair: { x: 150, y: 10, width: 200 },
    eyebrows: { x: 190, y: 110, width: 120 },
    eyes: { x: 190, y: 130, width: 120 },
    nose: { x: 220, y: 170, width: 60 },
    lips: { x: 200, y: 210, width: 100 },
    mustache: { x: 200, y: 190, width: 100 },
    ears: { x: 120, y: 130, width: 260 },
    neck: { x: 200, y: 250, width: 120 }
  };

  const [{ isOver }, drop] = useDrop(() => ({
    accept: "FACE_PART",

    drop: (item) => {

      setHistory(prev => [...prev, elements]);

      setElements(prev => {

        if (item.part === "ears") {

          const earCount = Object.keys(prev).filter(key =>
            key.startsWith("ears")
          ).length;

          const newKey = `ears${earCount + 1}`;

          return {
            ...prev,
            [newKey]: {
              src: item.src,
              ...defaultPositions[item.part]
            }
          };
        }

        return {
          ...prev,
          [item.part]: {
            src: item.src,
            ...defaultPositions[item.part]
          }
        };
      });
    },

    collect: monitor => ({
      isOver: monitor.isOver()
    })

  }), [elements]);

  return (
    <div className="canvas-area">

      <div
        ref={drop}
        id="canvas-box"
        className="canvas-box"
        style={{
          position: "relative",
          width: "500px",
          height: "500px",
          border: isOver ? "2px solid green" : "2px dashed gray"
        }}
      >

        {Object.keys(elements).map((part) => {

          const element = elements[part];

          return (
            <Rnd
              key={part}
              size={{ width: element.width, height: "auto" }}
              position={{ x: element.x, y: element.y }}

              onDragStop={(e, d) => {

                setHistory(prev => [...prev, elements]);

                setElements(prev => ({
                  ...prev,
                  [part]: { ...prev[part], x: d.x, y: d.y }
                }));
              }}

              onResizeStop={(e, direction, ref, delta, position) => {

                setHistory(prev => [...prev, elements]);

                setElements(prev => ({
                  ...prev,
                  [part]: {
                    ...prev[part],
                    width: ref.offsetWidth,
                    ...position
                  }
                }));
              }}
            >

              <img
                src={element.src}
                alt={part}
                style={{
                  width: "100%",
                  cursor: "pointer",
                  border: selectedPart === part ? "2px solid red" : "none"
                }}
                onClick={() => setSelectedPart(part)}
              />

            </Rnd>
          );
        })}

      </div>

    </div>
  );
}

export default CanvasArea;