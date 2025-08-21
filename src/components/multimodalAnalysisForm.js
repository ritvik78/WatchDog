import React, { useState, useRef } from 'react';
import PropTypes from 'prop-types';
import { InvokeLLM } from '@/integrations/Core';
import { Detection } from '@/entities/Detection';
import { Button } from '@/components/ui/button';
import { Textarea } from '@/components/ui/textarea';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from '@/components/ui/input';
import { Bot, Loader2, AlertTriangle, CheckCircle, Search, Mic, Volume2, FileAudio } from 'lucide-react';
import { Badge } from '@/components/ui/badge';

export default function MultimodalAnalysisForm({ onAnalysisComplete }) {
  const [inputText, setInputText] = useState('');
  const [inputUrl, setInputUrl] = useState('');
  const [audioFile, setAudioFile] = useState(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [result, setResult] = useState(null);
  const audioInputRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const [recordedAudio, setRecordedAudio] = useState(null);

  const handleTextAnalysis = async (content, platform) => {
    const prompt = `You are the 'WatchDog' verbal abuse detection system. Analyze the following text for verbal abuse. The text: "${content}". Respond with JSON: { "is_abusive": boolean, "confidence_score": number (0.0 to 1.0), "detection_type": string ('Explicit', 'Sarcasm', 'Hate Speech', 'Aggressive Tone', or 'None'), "language": string ('Hindi', 'Punjabi', 'Hinglish', 'English', or 'Other') }`;
    return await InvokeLLM({
      prompt,
      response_json_schema: {
        type: 'object',
        properties: {
          is_abusive: { type: 'boolean' },
          confidence_score: { type: 'number' },
          detection_type: { type: 'string', enum: ['Explicit', 'Sarcasm', 'Hate Speech', 'Aggressive Tone', 'None'] },
          language: { type: 'string' }
        }
      }
    });
  };

  const handleAudioAnalysis = async (audioContent) => {
    const prompt = `Analyze this audio content for aggressive speech patterns, tone modulation, and emotional indicators that might suggest abusive language. Simulate audio analysis results with JSON: { "pitch_variation": number (0.0 to 1.0), "tone_aggressiveness": number (0.0 to 1.0), "speaking_rate": number (100-300), "emotion_detected": string ('Neutral', 'Angry', 'Aggressive', 'Sarcastic', 'Happy', 'Sad'), "audio_confidence": number (0.0 to 1.0) }`;
    return await InvokeLLM({
      prompt,
      response_json_schema: {
        type: 'object',
        properties: {
          pitch_variation: { type: 'number' },
          tone_aggressiveness: { type: 'number' },
          speaking_rate: { type: 'number' },
          emotion_detected: { type: 'string', enum: ['Neutral', 'Angry', 'Aggressive', 'Sarcastic', 'Happy', 'Sad'] },
          audio_confidence: { type: 'number' }
        }
      }
    });
  };

  const handleMultimodalAnalysis = async (content, platform, analysisMode) => {
    setIsLoading(true);
    setResult(null);

    try {
      let textAnalysis = null;
      let audioAnalysis = null;
      let finalConfidence = 0;
      let isAbusive = false;

      if (analysisMode === 'Text Only' || analysisMode === 'Multimodal (Text + Audio)') {
        textAnalysis = await handleTextAnalysis(content, platform);
      }

      if (analysisMode === 'Audio Only' || analysisMode === 'Multimodal (Text + Audio)') {
        audioAnalysis = await handleAudioAnalysis(content);
      }

      if (analysisMode === 'Multimodal (Text + Audio)' && textAnalysis && audioAnalysis) {
        finalConfidence = (textAnalysis.confidence_score * 0.6) + (audioAnalysis.audio_confidence * 0.4);
        isAbusive = textAnalysis.is_abusive || audioAnalysis.tone_aggressiveness > 0.7;
      } else if (analysisMode === 'Text Only' && textAnalysis) {
        finalConfidence = textAnalysis.confidence_score;
        isAbusive = textAnalysis.is_abusive;
      } else if (analysisMode === 'Audio Only' && audioAnalysis) {
        finalConfidence = audioAnalysis.audio_confidence;
        isAbusive = audioAnalysis.tone_aggressiveness > 0.6;
      }

      const detectionResult = {
        status: isAbusive ? 'abusive' : 'safe',
        confidence_score: finalConfidence,
        detection_type: textAnalysis?.detection_type || 'Aggressive Tone',
        language: textAnalysis?.language || 'English',
        analysis_mode: analysisMode,
        text_confidence: textAnalysis?.confidence_score ?? null,
        audio_confidence: audioAnalysis?.audio_confidence ?? null,
        audio_features: audioAnalysis ? {
          pitch_variation: audioAnalysis.pitch_variation,
          tone_aggressiveness: audioAnalysis.tone_aggressiveness,
          speaking_rate: audioAnalysis.speaking_rate,
          emotion_detected: audioAnalysis.emotion_detected
        } : null
      };

      setResult(detectionResult);

      if (isAbusive) {
        const newDetection = await Detection.create({
          content: content,
          source_url: platform === 'Manual Input' || platform === 'Audio Upload' ? null : content,
          platform: platform,
          confidence_score: finalConfidence,
          detection_type: detectionResult.detection_type,
          language: detectionResult.language,
          analysis_mode: analysisMode,
          text_confidence: detectionResult.text_confidence,
          audio_confidence: detectionResult.audio_confidence,
          audio_features: detectionResult.audio_features
        });
        if (onAnalysisComplete) onAnalysisComplete(newDetection);
      }
    } catch (error) {
      console.error("Analysis failed:", error);
      setResult({ status: 'error' });
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = (e, type) => {
    e.preventDefault();
    if (type === 'text' && inputText) {
      handleMultimodalAnalysis(inputText, 'Manual Input', 'Text Only');
    } else if (type === 'url' && inputUrl) {
      handleMultimodalAnalysis(inputUrl, 'YouTube', 'Multimodal (Text + Audio)');
    } else if (type === 'audio' && (audioFile || recordedAudio)) {
      const audioName = audioFile ? audioFile.name : 'Recorded Audio';
      handleMultimodalAnalysis(audioName, 'Audio Upload', 'Audio Only');
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaRecorderRef.current = new window.MediaRecorder(stream);
      const chunks = [];

      mediaRecorderRef.current.ondataavailable = (event) => chunks.push(event.data);
      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunks, { type: 'audio/wav' });
        setRecordedAudio(blob);
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (error) {
      console.error('Error accessing microphone:', error);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach(track => track.stop());
      setIsRecording(false);
    }
  };

  return (
    <Card className="dark:bg-gray-950">
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Bot /> Multimodal Analysis
        </CardTitle>
        <CardDescription>
          Test both text-based and voice-based abuse detection capabilities.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs defaultValue="text">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="text">Text Analysis</TabsTrigger>
            <TabsTrigger value="audio">Voice Analysis</TabsTrigger>
            <TabsTrigger value="url">Multimodal URL</TabsTrigger>
          </TabsList>
          
          <TabsContent value="text">
            <form onSubmit={(e) => handleSubmit(e, 'text')} className="space-y-4 pt-4">
              <Textarea
                placeholder="Enter text to analyze for abusive language..."
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                className="min-h-[100px] dark:bg-gray-900"
              />
              <Button type="submit" disabled={isLoading || !inputText}>
                {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                Analyze Text
              </Button>
            </form>
          </TabsContent>

          <TabsContent value="audio">
            <div className="space-y-4 pt-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <label className="text-sm font-medium">Upload Audio File</label>
                  <Input
                    ref={audioInputRef}
                    type="file"
                    accept="audio/*"
                    onChange={(e) => setAudioFile(e.target.files[0])}
                    className="dark:bg-gray-900"
                  />
                </div>
                <div className="space-y-2">
                  <label className="text-sm font-medium">Record Audio</label>
                  <div className="flex gap-2">
                    {!isRecording ? (
                      <Button type="button" onClick={startRecording} variant="outline">
                        <Mic className="mr-2 h-4 w-4" />
                        Start Recording
                      </Button>
                    ) : (
                      <Button type="button" onClick={stopRecording} variant="destructive">
                        <Volume2 className="mr-2 h-4 w-4" />
                        Stop Recording
                      </Button>
                    )}
                  </div>
                </div>
              </div>
              {(audioFile || recordedAudio) && (
                <div className="p-4 border rounded-lg dark:border-gray-700">
                  <div className="flex items-center gap-2">
                    <FileAudio className="h-4 w-4" />
                    <span className="text-sm">
                      {audioFile ? audioFile.name : 'Recorded Audio'} ready for analysis
                    </span>
                  </div>
                </div>
              )}
              <Button 
                onClick={(e) => handleSubmit(e, 'audio')} 
                disabled={isLoading || (!audioFile && !recordedAudio)}
              >
                {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                Analyze Audio
              </Button>
            </div>
          </TabsContent>

          <TabsContent value="url">
            <form onSubmit={(e) => handleSubmit(e, 'url')} className="space-y-4 pt-4">
              <Input
                placeholder="Enter YouTube or Instagram URL for multimodal analysis..."
                value={inputUrl}
                onChange={(e) => setInputUrl(e.target.value)}
                className="dark:bg-gray-900"
              />
              <div className="text-sm text-gray-500 dark:text-gray-400">
                This will analyze both audio and text content from the video.
              </div>
              <Button type="submit" disabled={isLoading || !inputUrl}>
                {isLoading ? <Loader2 className="mr-2 h-4 w-4 animate-spin" /> : <Search className="mr-2 h-4 w-4" />}
                Multimodal Analysis
              </Button>
            </form>
          </TabsContent>
        </Tabs>
        
        {result && (
          <div className="mt-6 p-4 rounded-lg border dark:border-gray-700 space-y-3">
            {result.status === 'abusive' && (
              <div className="flex items-center gap-3 text-red-600 dark:text-red-400">
                <AlertTriangle className="h-6 w-6" />
                <div>
                  <p className="font-bold">Abusive Content Detected</p>
                  <div className="flex gap-2 mt-2">
                    <Badge variant="destructive">{result.detection_type}</Badge>
                    <Badge variant="outline">{result.analysis_mode}</Badge>
                    <Badge variant="secondary">{result.language}</Badge>
                  </div>
                </div>
              </div>
            )}
            {result.status === 'safe' && (
              <div className="flex items-center gap-3 text-green-600 dark:text-green-400">
                <CheckCircle className="h-6 w-6" />
                <div>
                  <p className="font-bold">Content Appears Safe</p>
                  <Badge variant="outline" className="mt-2">{result.analysis_mode}</Badge>
                </div>
              </div>
            )}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 text-sm">
              {result.text_confidence !== null && (
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
                  <p className="font-medium">Text Analysis</p>
                  <p>Confidence: {(result.text_confidence * 100).toFixed(0)}%</p>
                </div>
              )}
              {result.audio_confidence !== null && (
                <div className="p-3 bg-gray-50 dark:bg-gray-800 rounded">
                  <p className="font-medium">Audio Analysis</p>
                  <p>Confidence: {(result.audio_confidence * 100).toFixed(0)}%</p>
                  {result.audio_features && (
                    <div className="mt-2 space-y-1">
                      <p>Emotion: {result.audio_features.emotion_detected}</p>
                      <p>Tone Aggressiveness: {(result.audio_features.tone_aggressiveness * 100).toFixed(0)}%</p>
                    </div>
                  )}
                </div>
              )}
            </div>
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 rounded">
              <p className="font-medium text-blue-800 dark:text-blue-300">Final Combined Score</p>
              <p className="text-blue-600 dark:text-blue-400">{(result.confidence_score * 100).toFixed(0)}% confidence</p>
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}

MultimodalAnalysisForm.propTypes = {
  onAnalysisComplete: PropTypes.func,
};
