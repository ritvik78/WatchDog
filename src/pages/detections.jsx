
import React, { useState, useEffect } from 'react';
import { Detection } from '@/entities/Detection';
import DetectionsTable from '../components/DetectionsTable';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { debounce } from 'lodash';

export default function DetectionsPage() {
  const [allDetections, setAllDetections] = useState([]);
  const [filteredDetections, setFilteredDetections] = useState([]);
  const [filters, setFilters] = useState({
    search: '',
    status: 'all',
    platform: 'all'
  });

  const loadDetections = async () => {
    const detections = await Detection.list('-created_date');
    setAllDetections(detections);
    setFilteredDetections(detections);
  };

  useEffect(() => {
    loadDetections();
  }, []);

  const debouncedFilter = debounce(() => {
    let tempDetections = allDetections;

    if (filters.search) {
      tempDetections = tempDetections.filter(d => 
        d.content.toLowerCase().includes(filters.search.toLowerCase())
      );
    }
    if (filters.status !== 'all') {
      tempDetections = tempDetections.filter(d => d.status === filters.status);
    }
    if (filters.platform !== 'all') {
      tempDetections = tempDetections.filter(d => d.platform === filters.platform);
    }

    setFilteredDetections(tempDetections);
  }, 300);

  useEffect(() => {
    debouncedFilter();
    return () => debouncedFilter.cancel();
  }, [filters, allDetections]);
  
  const handleFilterChange = (key, value) => {
    setFilters(prev => ({...prev, [key]: value}));
  };

  const updateDetectionStatus = async (detection, newStatus) => {
    await Detection.update(detection.id, { status: newStatus });
    loadDetections();
  };

  return (
    <div className="space-y-6">
        <div>
            <h1 className="text-3xl font-bold text-gray-900 dark:text-white">All Detections</h1>
            <p className="text-gray-500 dark:text-gray-400">Search, filter, and manage all flagged content.</p>
        </div>

        <div className="flex flex-col sm:flex-row gap-4">
            <Input 
                placeholder="Search content..."
                value={filters.search}
                onChange={(e) => handleFilterChange('search', e.target.value)}
                className="max-w-sm dark:bg-gray-900"
            />
            <Select value={filters.status} onValueChange={(value) => handleFilterChange('status', value)}>
                <SelectTrigger className="w-full sm:w-[180px] dark:bg-gray-900">
                    <SelectValue placeholder="Filter by Status" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="all">All Statuses</SelectItem>
                    <SelectItem value="Flagged">Flagged</SelectItem>
                    <SelectItem value="Reviewed - Safe">Reviewed - Safe</SelectItem>
                    <SelectItem value="Reviewed - Abusive">Reviewed - Abusive</SelectItem>
                </SelectContent>
            </Select>
            <Select value={filters.platform} onValueChange={(value) => handleFilterChange('platform', value)}>
                <SelectTrigger className="w-full sm:w-[180px] dark:bg-gray-900">
                    <SelectValue placeholder="Filter by Platform" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="all">All Platforms</SelectItem>
                    <SelectItem value="YouTube">YouTube</SelectItem>
                    <SelectItem value="Instagram">Instagram</SelectItem>
                    <SelectItem value="Twitter">Twitter</SelectItem>
                    <SelectItem value="Manual Input">Manual Input</SelectItem>
                    <SelectItem value="Other">Other</SelectItem>
                </SelectContent>
            </Select>
        </div>

        <DetectionsTable
            detections={filteredDetections}
            title="Detections Log"
            onUpdateStatus={updateDetectionStatus}
        />
    </div>
  );
}
