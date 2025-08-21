
import React, { useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { X } from 'lucide-react';

export default function SettingsPage() {
    const [sensitivity, setSensitivity] = useState(75);
    const [keywords, setKeywords] = useState(['example-keyword', 'another-term']);
    const [newKeyword, setNewKeyword] = useState('');

    const addKeyword = (e) => {
        e.preventDefault();
        if(newKeyword && !keywords.includes(newKeyword)){
            setKeywords([...keywords, newKeyword]);
            setNewKeyword('');
        }
    };

    const removeKeyword = (keywordToRemove) => {
        setKeywords(keywords.filter(kw => kw !== keywordToRemove));
    };

    return (
        <div className="space-y-6 max-w-2xl">
            <div>
                <h1 className="text-3xl font-bold text-gray-900 dark:text-white">Settings</h1>
                <p className="text-gray-500 dark:text-gray-400">Configure the detection model and system preferences.</p>
            </div>

            <Card className="dark:bg-gray-950">
                <CardHeader>
                    <CardTitle>Detection Sensitivity</CardTitle>
                    <CardDescription>Adjust the confidence threshold for flagging content. Higher sensitivity means more content may be flagged.</CardDescription>
                </CardHeader>
                <CardContent>
                    <div className="flex items-center gap-4">
                        <Slider
                            value={[sensitivity]}
                            onValueChange={(value) => setSensitivity(value[0])}
                            max={100}
                            step={1}
                        />
                        <span className="font-semibold w-12 text-center">{sensitivity}%</span>
                    </div>
                </CardContent>
            </Card>

            <Card className="dark:bg-gray-950">
                <CardHeader>
                    <CardTitle>Managed Keywords</CardTitle>
                    <CardDescription>Add or remove specific keywords to always flag or ignore.</CardDescription>
                </CardHeader>
                <CardContent>
                    <form onSubmit={addKeyword} className="flex gap-2 mb-4">
                        <Input
                            placeholder="Add a new keyword..."
                            value={newKeyword}
                            onChange={(e) => setNewKeyword(e.target.value)}
                            className="dark:bg-gray-900"
                        />
                        <Button type="submit">Add</Button>
                    </form>
                    <div className="flex flex-wrap gap-2">
                        {keywords.map(keyword => (
                            <Badge key={keyword} variant="secondary" className="pl-3 pr-1 py-1 text-sm dark:bg-gray-800">
                                {keyword}
                                <button onClick={() => removeKeyword(keyword)} className="ml-1 rounded-full p-0.5 hover:bg-red-200 dark:hover:bg-red-900/50">
                                    <X className="w-3 h-3"/>
                                </button>
                            </Badge>
                        ))}
                    </div>
                </CardContent>
            </Card>
            
             <div className="flex justify-end">
                <Button>Save Changes</Button>
            </div>
        </div>
    );
}
