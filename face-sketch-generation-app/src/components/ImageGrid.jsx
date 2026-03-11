import React from "react";
import { useDrag } from "react-dnd";

function DraggableImage({ src, part }) {

  const [{ isDragging }, drag] = useDrag(() => ({
    type: "FACE_PART",
    item: { src: src, part: part },
    collect: (monitor) => ({
      isDragging: monitor.isDragging()
    })
  }), [src, part]);

  return (
    <div
      ref={drag}
      className="thumb"
      style={{
        opacity: isDragging ? 0.5 : 1,
        cursor: "grab"
      }}
    >
      <img src={src} alt={part} />
    </div>
  );
}

function ImageGrid({ part }) {

  const partLengths = {
    head: 10,
    hair: 12,
    eyes: 12,
    eyebrows: 12,
    nose: 12,
    lips: 12,
    mustache: 12,
    ears: 4,
    neck: 2
  };

  const totalImages = partLengths[part] || 0;

  const images = [];

  for (let i = 1; i <= totalImages; i++) {
    images.push(`/Images/${part}/${i}.png`);
  }

  return (
    <div className="image-grid">
      {images.map((src, index) => (
        <DraggableImage
          key={index}
          src={src}
          part={part}
        />
      ))}
    </div>
  );
}

export default ImageGrid;