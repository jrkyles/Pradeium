import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ProbabilityPoint } from "../types/api";

type ProbabilityChartProps = {
  data: ProbabilityPoint[];
  comparison?: {
    data: { category: string; current: number; previous: number }[];
    betterIsCurrent: boolean;
  };
};

export const ProbabilityChart = ({ data, comparison }: ProbabilityChartProps) => {
  if (comparison) {
    const currentColor = comparison.betterIsCurrent ? "#6c3fd6" : "#b0b4c3";
    const previousColor = comparison.betterIsCurrent ? "#b0b4c3" : "#6c3fd6";
    return (
      <div className="chart-wrapper">
        <ResponsiveContainer width="100%" height={360}>
          <BarChart data={comparison.data}>
            <XAxis dataKey="category" />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(val) => `${Math.round(val * 100)}%`}
            />
            <Tooltip
              formatter={(value: number) => `${(value * 100).toFixed(1)}%`}
              labelFormatter={(label) => `Category: ${label}`}
            />
            <Bar
              dataKey="previous"
              name="Previous prediction"
              fill={previousColor}
              radius={[10, 10, 0, 0]}
            />
            <Bar
              dataKey="current"
              name="Current prediction"
              fill={currentColor}
              radius={[10, 10, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
      </div>
    );
  }

  return (
    <div className="chart-wrapper">
      <ResponsiveContainer width="100%" height={360}>
        <BarChart data={data}>
          <XAxis dataKey="category" />
          <YAxis
            domain={[0, 1]}
            tickFormatter={(val) => `${Math.round(val * 100)}%`}
          />
          <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
          <Bar dataKey="probability" fill="#6c3fd6" radius={[12, 12, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};
