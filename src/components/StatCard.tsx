import React from 'react';

type StatCardProps = {
  title: string;
  value: string | number;
  icon?: React.ElementType;
  description?: string;
};

export function StatCard({ title, value, icon: Icon, description }: StatCardProps) {
  return (
    <div className="p-4 border rounded">
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-500">{title}</div>
        {Icon && <Icon className="h-4 w-4" />}
      </div>
      <div className="text-2xl font-bold">{value}</div>
      {description && <p className="text-xs text-gray-500">{description}</p>}
    </div>
  );
}
