import React from "react";
import "../styles/Sidebar.css";

const parts = [
  "Head",
  "Hair",
  "Eyes",
  "Eyebrows",
  "Nose",
  "Lips",
  "Mustache",
  "Ears",
  "Neck"
];

function Sidebar({ setSelectedPart }) {
  return (
    <div className="sidebar">
      {parts.map((part) => {
        const lowerPart = part.toLowerCase();
        const imagePath = `/Images/${lowerPart}/1.png`;

        return (
          <div
            key={part}
            className="sidebar-item"
            onClick={() => setSelectedPart(lowerPart)}
          >
            <div className="icon-circle">
              <img
                src={imagePath}
                alt={part}
                className="icon-image"
              />
            </div>
            <span>{part}</span>
          </div>
        );
      })}
    </div>
  );
}

export default Sidebar;