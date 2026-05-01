'use client';

export default function TrajectoriesPage() {
  return (
    <div className="w-full h-screen">
      <iframe
        src="/globe.html"
        className="w-full h-full border-0"
        title="SSIS Globe Visualization"
      />
    </div>
  );
}
