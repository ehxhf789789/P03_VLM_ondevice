interface Props {
  src: string;
  alt?: string;
  className?: string;
}

export function ImagePreview({ src, alt = "Preview", className = "" }: Props) {
  return (
    <div
      className={`relative overflow-hidden rounded-2xl bg-gray-100 ${className}`}
    >
      <img
        src={src}
        alt={alt}
        className="w-full h-full object-cover"
        draggable={false}
      />
    </div>
  );
}
