'use client';

import React, { useState, useEffect } from 'react';
import {
  SettingsIcon,
  ChevronDownIcon,
  CheckIcon,
  XIcon,
  RotateCcwIcon,
  ZapIcon,
  SearchIcon,
  BrainIcon,
  FilterIcon,
  EyeIcon
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Slider } from '@/components/ui/slider';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuTrigger,
  DropdownMenuLabel,
  DropdownMenuSeparator,
} from '@/components/ui/dropdown-menu';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Badge } from '@/components/ui/badge';
import { ChatSettings, DEFAULT_CHAT_SETTINGS, SETTINGS_PRESETS } from '@/types/chat';

interface ChatSettingsDropdownProps {
  currentSettings: ChatSettings;
  onSettingsUpdate: (settings: ChatSettings) => Promise<void>;
  isUpdating?: boolean;
}

export function ChatSettingsDropdown({
  currentSettings,
  onSettingsUpdate,
  isUpdating = false
}: ChatSettingsDropdownProps) {
  const [localSettings, setLocalSettings] = useState<ChatSettings>(currentSettings);
  const [isOpen, setIsOpen] = useState(false);
  const [hasChanges, setHasChanges] = useState(false);

  // Update local settings when current settings change
  useEffect(() => {
    setLocalSettings(currentSettings);
  }, [currentSettings]);

  // Check if there are unsaved changes
  useEffect(() => {
    const changed = JSON.stringify(localSettings) !== JSON.stringify(currentSettings);
    setHasChanges(changed);
  }, [localSettings, currentSettings]);

  const handleSettingChange = (key: keyof ChatSettings, value: any) => {
    setLocalSettings(prev => ({
      ...prev,
      [key]: value
    }));
  };

  const handleSave = async () => {
    try {
      await onSettingsUpdate(localSettings);
      setIsOpen(false);
    } catch (error) {
      console.error('Failed to update settings:', error);
    }
  };

  const handleReset = () => {
    setLocalSettings(currentSettings);
  };

  const handlePresetSelect = (presetKey: string) => {
    const preset = SETTINGS_PRESETS[presetKey as keyof typeof SETTINGS_PRESETS];
    if (preset) {
      setLocalSettings(preset);
    }
  };

  const getTreeLevelDescription = (levels: number[] | null) => {
    if (!levels || levels.length === 0) return 'All levels';
    if (levels.includes(0) && levels.length === 1) return 'Original only';
    if (levels.every(l => l > 0)) return 'Summaries only';
    return `Levels: ${levels.join(', ')}`;
  };

  return (
    <DropdownMenu open={isOpen} onOpenChange={setIsOpen}>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          size="sm"
          className="h-9 w-9 p-0 relative"
        >
          <SettingsIcon className="h-4 w-4" />
          {hasChanges && (
            <div className="absolute -top-1 -right-1 h-2 w-2 bg-orange-500 rounded-full" />
          )}
        </Button>
      </DropdownMenuTrigger>
      
      <DropdownMenuContent 
        align="start" 
        className="w-96 p-4 max-h-[600px] overflow-y-auto"
        side="top"
      >
        <DropdownMenuLabel className="flex items-center gap-2 pb-2">
          <SettingsIcon className="h-4 w-4" />
          Chat Settings
        </DropdownMenuLabel>
        
        <DropdownMenuSeparator />

        {/* Quick Presets */}
        <div className="space-y-2 py-2">
          <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <ZapIcon className="h-3 w-3" />
            Quick Presets
          </Label>
          <div className="flex flex-wrap gap-1">
            {Object.entries(SETTINGS_PRESETS).map(([key, preset]) => (
              <Badge
                key={key}
                variant="outline"
                className="cursor-pointer hover:bg-accent text-xs"
                onClick={() => handlePresetSelect(key)}
              >
                {key.charAt(0).toUpperCase() + key.slice(1).replace('_', ' ')}
              </Badge>
            ))}
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Search Method Settings */}
        <div className="space-y-3 py-2">
          <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <SearchIcon className="h-3 w-3" />
            Search Method
          </Label>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm">Hierarchical Search</Label>
                <p className="text-xs text-muted-foreground">Use document summaries for better context</p>
              </div>
              <Switch
                checked={localSettings.use_tree_search}
                onCheckedChange={(checked) => handleSettingChange('use_tree_search', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <div className="space-y-0.5">
                <Label className="text-sm">Hybrid Search</Label>
                <p className="text-xs text-muted-foreground">Combine semantic + keyword matching</p>
              </div>
              <Switch
                checked={localSettings.use_hybrid_search}
                onCheckedChange={(checked) => handleSettingChange('use_hybrid_search', checked)}
              />
            </div>

            {localSettings.use_hybrid_search && (
              <div className="space-y-2">
                <Label className="text-sm">Vector Weight: {localSettings.vector_weight.toFixed(1)}</Label>
                <div className="px-2">
                  <Slider
                    value={[localSettings.vector_weight]}
                    onValueChange={([value]) => handleSettingChange('vector_weight', value)}
                    max={1}
                    min={0}
                    step={0.1}
                    className="w-full"
                  />
                  <div className="flex justify-between text-xs text-muted-foreground mt-1">
                    <span>Keyword</span>
                    <span>Semantic</span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Retrieval Settings */}
        <div className="space-y-3 py-2">
          <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <FilterIcon className="h-3 w-3" />
            Retrieval Settings
          </Label>
          
          <div className="space-y-3">
            <div className="space-y-2">
              <Label className="text-sm">Sources to Retrieve: {localSettings.top_k}</Label>
              <Slider
                value={[localSettings.top_k]}
                onValueChange={([value]) => handleSettingChange('top_k', value)}
                max={50}
                min={1}
                step={1}
                className="w-full"
              />
              <div className="flex justify-between text-xs text-muted-foreground">
                <span>1</span>
                <span>50</span>
              </div>
            </div>

            <div className="space-y-2">
              <Label className="text-sm">Content Levels</Label>
              <Select
                value={
                  !localSettings.tree_level_filter ? 'all' :
                  localSettings.tree_level_filter.includes(0) && localSettings.tree_level_filter.length === 1 ? 'original' :
                  localSettings.tree_level_filter.every(l => l > 0) ? 'summaries' : 'custom'
                }
                onValueChange={(value) => {
                  switch (value) {
                    case 'all':
                      handleSettingChange('tree_level_filter', null);
                      break;
                    case 'original':
                      handleSettingChange('tree_level_filter', [0]);
                      break;
                    case 'summaries':
                      handleSettingChange('tree_level_filter', [1, 2, 3]);
                      break;
                    default:
                      break;
                  }
                }}
              >
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="all">All levels</SelectItem>
                  <SelectItem value="original">Original chunks only</SelectItem>
                  <SelectItem value="summaries">Summary levels only</SelectItem>
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground">
                {getTreeLevelDescription(localSettings.tree_level_filter)}
              </p>
            </div>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Model Settings */}
        <div className="space-y-3 py-2">
          <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <BrainIcon className="h-3 w-3" />
            Model Settings
          </Label>
          
          <div className="space-y-2">
            <Label className="text-sm">LLM Model</Label>
            <Input
              value={localSettings.llm_model || ''}
              onChange={(e) => handleSettingChange('llm_model', e.target.value || null)}
              placeholder="Default model (e.g., llama3.2)"
              className="text-sm"
            />
            <p className="text-xs text-muted-foreground">
              Leave empty to use default model
            </p>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* UI Preferences */}
        <div className="space-y-3 py-2">
          <Label className="text-xs font-medium text-muted-foreground flex items-center gap-1">
            <EyeIcon className="h-3 w-3" />
            Display Preferences
          </Label>
          
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label className="text-sm">Show Sources by Default</Label>
              <Switch
                checked={localSettings.show_sources}
                onCheckedChange={(checked) => handleSettingChange('show_sources', checked)}
              />
            </div>

            <div className="flex items-center justify-between">
              <Label className="text-sm">Auto-scroll to New Messages</Label>
              <Switch
                checked={localSettings.auto_scroll}
                onCheckedChange={(checked) => handleSettingChange('auto_scroll', checked)}
              />
            </div>
          </div>
        </div>

        <DropdownMenuSeparator />

        {/* Action Buttons */}
        <div className="flex items-center justify-between pt-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleReset}
            disabled={!hasChanges}
            className="flex items-center gap-1"
          >
            <RotateCcwIcon className="h-3 w-3" />
            Reset
          </Button>
          
          <div className="flex items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsOpen(false)}
            >
              <XIcon className="h-3 w-3" />
            </Button>
            <Button
              size="sm"
              onClick={handleSave}
              disabled={!hasChanges || isUpdating}
              className="flex items-center gap-1"
            >
              <CheckIcon className="h-3 w-3" />
              {isUpdating ? 'Updating...' : 'Update'}
            </Button>
          </div>
        </div>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}