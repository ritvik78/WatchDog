
import React from 'react';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { MoreVertical } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuTrigger } from '@/components/ui/dropdown-menu';
import { formatDistanceToNow } from 'date-fns';

export default function DetectionsTable({ detections, title, description, onUpdateStatus }) {
  
  const getStatusBadge = (status) => {
    switch (status) {
      case 'Flagged':
        return <Badge variant="destructive">Flagged</Badge>;
      case 'Reviewed - Safe':
        return <Badge variant="secondary" className="bg-green-100 text-green-800 dark:bg-green-900/50 dark:text-green-400">Safe</Badge>;
      case 'Reviewed - Abusive':
        return <Badge className="bg-yellow-100 text-yellow-800 dark:bg-yellow-900/50 dark:text-yellow-400">Abusive</Badge>;
      default:
        return <Badge>{status}</Badge>;
    }
  };

  return (
    <Card className="dark:bg-gray-950">
      <CardHeader>
        <CardTitle>{title}</CardTitle>
        {description && <CardDescription>{description}</CardDescription>}
      </CardHeader>
      <CardContent>
        <div className="border rounded-lg dark:border-gray-800">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Content</TableHead>
                <TableHead>Confidence</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Platform</TableHead>
                <TableHead>Detected</TableHead>
                <TableHead className="text-right">Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {detections && detections.length > 0 ? detections.map((detection) => (
                <TableRow key={detection.id}>
                  <TableCell className="font-medium max-w-xs truncate">
                    <span className="text-gray-800 dark:text-gray-200">{detection.content}</span>
                  </TableCell>
                  <TableCell>
                    <span className="text-red-500 font-semibold">{(detection.confidence_score * 100).toFixed(0)}%</span>
                  </TableCell>
                  <TableCell>{getStatusBadge(detection.status)}</TableCell>
                  <TableCell>
                    <Badge variant="outline" className="dark:border-gray-700">{detection.platform}</Badge>
                  </TableCell>
                  <TableCell className="text-gray-500 dark:text-gray-400 text-xs">
                    {formatDistanceToNow(new Date(detection.created_date), { addSuffix: true })}
                  </TableCell>
                  <TableCell className="text-right">
                    <DropdownMenu>
                      <DropdownMenuTrigger asChild>
                        <Button variant="ghost" size="icon">
                          <MoreVertical className="w-4 h-4" />
                        </Button>
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem onClick={() => onUpdateStatus && onUpdateStatus(detection, 'Reviewed - Safe')}>Mark as Safe</DropdownMenuItem>
                        <DropdownMenuItem onClick={() => onUpdateStatus && onUpdateStatus(detection, 'Reviewed - Abusive')}>Mark as Abusive</DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </TableCell>
                </TableRow>
              )) : (
                <TableRow>
                    <TableCell colSpan={6} className="text-center h-24 text-gray-500 dark:text-gray-400">
                        No detections found.
                    </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </div>
      </CardContent>
    </Card>
  );
}
