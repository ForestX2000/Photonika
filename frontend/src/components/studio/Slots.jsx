import { useRef, useState } from "react";
import { zLabel } from "../../lib/format";

function Slot({ index, slot, distance, onPick, onDropFiles, onRemove }) {
  const [drag, setDrag] = useState(false);
  const label = `${index + 1} · ${zLabel(distance)}`;
  return (
    <div
      className={`slot${slot ? " filled" : ""}${drag ? " drag" : ""}`}
      tabIndex={0}
      role="button"
      aria-label={`Изображение ${index + 1}, z = ${zLabel(distance)}${slot ? ", загружено" : ", пусто"}`}
      onClick={() => onPick(index)}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          onPick(index);
        }
      }}
      onDragOver={(e) => {
        e.preventDefault();
        setDrag(true);
      }}
      onDragLeave={() => setDrag(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDrag(false);
        onDropFiles([...e.dataTransfer.files].filter((f) => f.type.startsWith("image/")), index);
      }}
    >
      {slot ? (
        <>
          <img alt="" src={slot.url} />
          <span className="slot-tag">{label}</span>
          <button
            className="slot-remove"
            type="button"
            aria-label={`Убрать изображение ${index + 1}`}
            onClick={(e) => {
              e.stopPropagation();
              onRemove(index);
            }}
          >
            ×
          </button>
        </>
      ) : (
        <>
          <span className="slot-plus">+</span>
          <span className="slot-z">{label}</span>
          <span className="slot-hint">перетащите или нажмите</span>
        </>
      )}
    </div>
  );
}

/** Четыре ячейки под изображения: клик — выбор файла, перетаскивание одного или нескольких файлов. */
export default function Slots({ slots, distances, onSetFile, onDistribute, onRemove, onClear }) {
  const fileInput = useRef(null);
  const multiInput = useRef(null);
  const pickIndex = useRef(0);

  const pick = (i) => {
    pickIndex.current = i;
    fileInput.current.value = "";
    fileInput.current.click();
  };
  const dropFiles = (files, i) => {
    if (files.length > 1) onDistribute(files, i);
    else if (files[0]) onSetFile(i, files[0]);
  };

  return (
    <>
      <div className="slots" id="slots">
        {slots.map((slot, i) => (
          <Slot key={i} index={i} slot={slot} distance={distances[i]} onPick={pick} onDropFiles={dropFiles} onRemove={onRemove} />
        ))}
      </div>
      <input
        type="file"
        id="fileInput"
        accept="image/*"
        hidden
        ref={fileInput}
        onChange={(e) => e.target.files[0] && onSetFile(pickIndex.current, e.target.files[0])}
      />
      <input
        type="file"
        id="multiInput"
        accept="image/*"
        multiple
        hidden
        ref={multiInput}
        onChange={(e) => e.target.files.length && onDistribute([...e.target.files], 0)}
      />
      <div className="slots-actions">
        <button
          className="btn btn-ghost btn-sm"
          type="button"
          id="pickAll"
          onClick={() => {
            multiInput.current.value = "";
            multiInput.current.click();
          }}
        >
          Загрузить 4 файла сразу
        </button>
        <button className="btn btn-link btn-sm" type="button" id="clearSlots" onClick={onClear}>
          Очистить
        </button>
      </div>
    </>
  );
}
