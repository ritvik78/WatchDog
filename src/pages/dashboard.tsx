import React, { useState, useEffect } from 'react';
import { AlertTriangle, ShieldCheck, Percent, ListChecks, Mic, FileText } from 'lucide-react';

// Component Types
type StatCardProps = {
  title: string;
  value: string | number;
  icon?: React.ElementType;
  description?: string;
};

type MultimodalFormProps = {
  onAnalysisComplete: (detection: DetectionType) => void;
};

type DetectionsTableProps = {
  title: string;
  detections: DetectionType[];
  onUpdateStatus: (detection: DetectionType, status: string) => void;
};

type ButtonProps = {
  asChild?: boolean;
  variant?: 'outline' | 'default';
  children: React.ReactNode;
} & React.ButtonHTMLAttributes<HTMLButtonElement>;

// Mock components with proper TypeScript types
function StatCard({ title, value, icon: Icon, description }: StatCardProps) {
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

function MultimodalAnalysisForm({ onAnalysisComplete }: MultimodalFormProps) {
  return (
    <div className="p-4 border rounded">
      <h3>Analysis Form</h3>
      <p>Component under development</p>
    </div>
  );
}

function DetectionsTable({ title, detections, onUpdateStatus }: DetectionsTableProps) {
  return (
    <div className="p-4 border rounded">
      <h3>{title}</h3>
      <div>Found {detections.length} detections</div>
    </div>
  );
}

function Button({ children, variant = 'default', className, ...props }: ButtonProps) {
  return (
    <button 
      className={`px-4 py-2 ${variant === 'outline' ? 'border border-gray-300' : 'bg-blue-500 text-white'} rounded ${className}`} 
      {...props}
    >
      {children}
    </button>
  );
}

function Link({ to, children, ...props }: { to: string; children: React.ReactNode }) {
  return <a href={to} {...props}>{children}</a>;
}
function StatCard({ title, value, icon: Icon, description }: StatCardProps) {
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

function MultimodalAnalysisForm({ onAnalysisComplete }: MultimodalFormProps) {
  return (
  <div className="p-4 border rounded">
    <h3>Analysis Form</h3>
    <p>Component under development</p>
  </div>
);

const DetectionsTable = ({ title, detections, onUpdateStatus }) => (
  <div className="p-4 border rounded">
    <h3>{title}</h3>
    <div>Found {detections.length} detections</div>
  </div>
);

const Button = ({ children, ...props }) => (
  <button className="px-4 py-2 bg-blue-500 text-white rounded" {...props}>
    {children}
  </button>
);

const Link = ({ to, children, ...props }) => (
  <a href={to} {...props}>{children}</a>
);

// --- Types ---
type DetectionType = {
  id: string;
  status: string;
  analysis_mode: string;
  confidence_score: number;
  platform: string;
  created_date: string;
  content: string;
};

type StatsType = {
  total: number;
  reviewed: number;
  textOnly: number;
  multimodal: number;
  accuracy: number;
  precision: number;
};

// --- Mock Data ---
const MOCK_DETECTIONS: DetectionType[] = [
  {
    id: '1',
    status: 'Flagged',
    analysis_mode: 'Text Only',
    confidence_score: 0.85,
    platform: 'Manual Input',
    created_date: new Date().toISOString(),
    content: 'Example abusive content'
  },
  {
    id: '2',
    status: 'Reviewed - Safe',
    analysis_mode: 'Multimodal (Text + Audio)',
    confidence_score: 0.65,
    platform: 'YouTube',
    created_date: new Date().toISOString(),
    content: 'Example safe content'
  }
];

// --- Dashboard ---
const Dashboard = () => {
  const [detections, setDetections] = useState<DetectionType[]>([]);
  const [stats, setStats] = useState<StatsType>({
    total: 0,
    reviewed: 0,
    textOnly: 0,
    multimodal: 0,
    accuracy: 78,
    precision: 82,
  });

  // Load mock data
  const loadData = async () => {
    const allDetections = MOCK_DETECTIONS;
    setDetections(allDetections);
    const total = allDetections.length;
    const reviewed = allDetections.filter(d => d.status !== 'Flagged').length;
    const textOnly = allDetections.filter(d => d.analysis_mode === 'Text Only').length;
    const multimodal = allDetections.filter(d => d.analysis_mode === 'Multimodal (Text + Audio)').length;
    setStats(prev => ({ ...prev, total, reviewed, textOnly, multimodal }));
  };

  useEffect(() => {
    loadData();
  }, []);

  const handleAnalysisComplete = (newDetection: DetectionType) => {
    setDetections(prev => [newDetection, ...prev]);
    setStats(prev => ({
      ...prev,
      total: prev.total + 1,
      textOnly: newDetection.analysis_mode === 'Text Only' ? prev.textOnly + 1 : prev.textOnly,
      multimodal: newDetection.analysis_mode === 'Multimodal (Text + Audio)' ? prev.multimodal + 1 : prev.multimodal,
    }));
  };

  const updateDetectionStatus = async (detection: DetectionType, newStatus: string) => {
    setDetections(prev =>
      prev.map(d =>
        d.id === detection.id ? { ...d, status: newStatus } : d
      )
    );
    loadData();
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-start">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">WATCHDOG Dashboard</h1>
          <p className="text-gray-500 dark:text-gray-400">Multimodal verbal abuse detection for Hindi, Punjabi & English content.</p>
        </div>
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Total Detections" value={stats.total} icon={AlertTriangle} description="All flagged content across platforms." />
        <StatCard title="Reviewed Items" value={stats.reviewed} icon={ListChecks} description={`${((stats.reviewed / (stats.total || 1)) * 100).toFixed(0)}% of detections reviewed.`} />
        <StatCard title="Text Analysis" value={stats.textOnly} icon={FileText} description="Text-only detections." />
        <StatCard title="Multimodal Analysis" value={stats.multimodal} icon={Mic} description="Combined text + audio analysis." />
      </div>

      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Model Accuracy" value={`${stats.accuracy}%`} icon={ShieldCheck} description="Based on test dataset (from research)." />
        <StatCard title="Model Precision" value={`${stats.precision}%`} icon={Percent} description="For abusive content detection." />
        <StatCard title="Recall Rate" value="74%" icon={ShieldCheck} description="From research evaluation." />
        <StatCard title="Processing Speed" value="1.2s" icon={AlertTriangle} description="Average analysis time per content." />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <MultimodalAnalysisForm onAnalysisComplete={handleAnalysisComplete} />
        <div className="min-h-[200px] lg:min-h-0">
          <DetectionsTable
            title="Recent Detections"
            detections={detections.slice(0, 5)}
            onUpdateStatus={updateDetectionStatus}
          />
          {detections.length > 5 && (
            <div className="text-right mt-4">
              <Button asChild variant="outline">
                <Link to="/detections">View All Detections</Link>
              </Button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export { Dashboard };