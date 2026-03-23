interface EloDisplayProps {
  score: number;
  size?: "sm" | "md" | "lg";
}

export default function EloDisplay({ score, size = "md" }: EloDisplayProps) {
  const tier = getEloTier(score);

  const sizeClasses = {
    sm: "text-lg",
    md: "text-3xl",
    lg: "text-5xl",
  };

  return (
    <div className="flex flex-col items-center gap-1">
      <span className={`${sizeClasses[size]} font-bold ${tier.color}`}>
        {score}
      </span>
      <span className={`text-xs font-medium uppercase tracking-wider ${tier.color}`}>
        {tier.label}
      </span>
    </div>
  );
}

function getEloTier(score: number): { label: string; color: string } {
  if (score >= 2000) return { label: "Grandmaster", color: "text-yellow-400" };
  if (score >= 1800) return { label: "Master", color: "text-purple-400" };
  if (score >= 1600) return { label: "Expert", color: "text-blue-400" };
  if (score >= 1400) return { label: "Advanced", color: "text-green-400" };
  if (score >= 1200) return { label: "Intermediate", color: "text-gray-300" };
  return { label: "Beginner", color: "text-gray-500" };
}
