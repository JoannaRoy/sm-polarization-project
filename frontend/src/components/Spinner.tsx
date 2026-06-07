export function Spinner({ label = "Loading..." }: { label?: string }) {
  return <div className="spinner">{label}</div>;
}
