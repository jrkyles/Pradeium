import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ProbabilityPoint } from "../types/api";

type ProbabilityChartProps = {
  data: ProbabilityPoint[];
};

export const ProbabilityChart = ({ data }: ProbabilityChartProps) => (
  <div className="chart-wrapper">
    <ResponsiveContainer width="100%" height={360}>
      <BarChart data={data}>
        <XAxis dataKey="category" />
        <YAxis
          domain={[0, 1]}
          tickFormatter={(val) => `${Math.round(val * 100)}%`}
        />
        <Tooltip formatter={(value: number) => `${(value * 100).toFixed(1)}%`} />
        <Bar dataKey="probability" fill="#0f3d2f" radius={[12, 12, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  </div>
);

