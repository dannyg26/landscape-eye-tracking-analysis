import os
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
from datetime import datetime

# Try to import pinyin library for name conversion
try:
    from pypinyin import lazy_pinyin, Style
    has_pinyin = True
except ImportError:
    has_pinyin = False
    print("Warning: pypinyin not installed. Names will not be converted to English.")

def pinyin_to_english(chinese_name):
    """Convert Chinese name to pinyin English transliteration."""
    if not has_pinyin:
        return chinese_name
    
    pinyin_list = lazy_pinyin(chinese_name, style=Style.NORMAL)
    # Capitalize first letter of each syllable
    english_name = ''.join([p.capitalize() for p in pinyin_list if p])
    return english_name

def extract_participants():
    """Extract all participant information."""
    participants_dir = r"c:\Users\danny\Downloads\tobi-project\Data\Participants"
    participants = {}
    
    for file in os.listdir(participants_dir):
        if file.endswith('.exp'):
            file_path = os.path.join(participants_dir, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Register namespace to handle defaults
                namespaces = {
                    '': 'http://schemas.datacontract.org/2004/07/Tobii.Studio.Project.Model.Variables.Entities',
                    'i': 'http://www.w3.org/2001/XMLSchema-instance'
                }
                
                # Extract Key (ID) and Name
                key_elem = root.find('.//Key')
                name_elem = root.find('.//Name')
                
                if key_elem is not None and name_elem is not None:
                    participant_id = key_elem.text
                    chinese_name = name_elem.text
                    english_name = pinyin_to_english(chinese_name)
                    
                    participants[participant_id] = {
                        'chinese_name': chinese_name,
                        'english_name': english_name,
                        'file': file
                    }
            except Exception as e:
                print(f"Error parsing {file}: {e}")
    
    return participants

def extract_variables():
    """Extract participant variables definitions."""
    variables_dir = r"c:\Users\danny\Downloads\tobi-project\Data\Variables"
    variables = {}
    
    for file in os.listdir(variables_dir):
        if file.endswith('.xml'):
            file_path = os.path.join(variables_dir, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                kind_elem = root.find('.//Kind')
                if kind_elem is not None and kind_elem.text == 'ParticipantVariable':
                    key_elem = root.find('.//Key')
                    name_elem = root.find('.//Name')
                    
                    if key_elem is not None and name_elem is not None:
                        var_id = key_elem.text
                        var_name = name_elem.text
                        
                        # Extract possible values
                        values = []
                        for value in root.findall('.//Value/Name'):
                            if value.text:
                                values.append(value.text)
                        
                        variables[var_id] = {
                            'name': var_name,
                            'values': values
                        }
            except Exception as e:
                print(f"Error parsing {file}: {e}")
    
    return variables

def map_recordings_to_participants():
    """Map recording IDs to participant IDs."""
    recordings_dir = r"c:\Users\danny\Downloads\tobi-project\Data\Recordings"
    recording_participant_map = {}
    
    for file in os.listdir(recordings_dir):
        if file.endswith('.rec'):
            file_path = os.path.join(recordings_dir, file)
            try:
                tree = ET.parse(file_path)
                root = tree.getroot()
                
                # Extract recording key (ID) and participant ID
                key_elem = root.find('.//Key')
                participant_id_elem = root.find('.//ParticipantId')
                
                if key_elem is not None and participant_id_elem is not None:
                    recording_id = key_elem.text
                    participant_id = participant_id_elem.text
                    recording_participant_map[recording_id] = participant_id
            except Exception as e:
                print(f"Error parsing {file}: {e}")
    
    return recording_participant_map

def get_unique_recordings_from_csv():
    """Get unique recording IDs from the attention CSV file."""
    csv_path = r"c:\Users\danny\Downloads\tobi-project\attention_by_recording.csv"
    recordings = set()
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if 'recording' in row:
                    recordings.add(row['recording'])
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return recordings

def create_student_csv():
    """Create a CSV file with student names and characteristics."""
    # Extract data
    print("Extracting participants...")
    participants = extract_participants()
    print(f"Found {len(participants)} participants")
    
    print("Extracting variables...")
    variables = extract_variables()
    print(f"Found {len(variables)} variables")
    
    print("Mapping recordings to participants...")
    recording_participant_map = map_recordings_to_participants()
    print(f"Found {len(recording_participant_map)} recordings")
    
    print("Getting unique recordings from CSV...")
    recordings_in_analysis = get_unique_recordings_from_csv()
    print(f"Found {len(recordings_in_analysis)} unique recordings in analysis")
    
    # Determine which participants are in the analysis
    active_participants = {}
    for recording_id in recordings_in_analysis:
        if recording_id in recording_participant_map:
            participant_id = recording_participant_map[recording_id]
            if participant_id in participants:
                active_participants[participant_id] = participants[participant_id]
    
    print(f"Found {len(active_participants)} active participants in analysis")
    
    # Create CSV output
    output_path = r"c:\Users\danny\Downloads\tobi-project\student_information.csv"
    
    try:
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Write header
            headers = ['Student Number', 'Chinese Name', 'English Name', 'Variable: Gender (性别)', 'Variable: Resolution']
            writer.writerow(headers)
            
            # Write participant data
            for idx, (participant_id, participant_data) in enumerate(sorted(active_participants.items()), 1):
                row = [
                    idx,
                    participant_data['chinese_name'],
                    participant_data['english_name'],
                    '',  # Gender - not yet available in current data structure
                    '',  # Resolution - not yet available
                ]
                writer.writerow(row)
        
        print(f"CSV file created: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error creating CSV: {e}")
        return None

if __name__ == '__main__':
    create_student_csv()
