"""Local human eye-state review. Shows source frames, never model predictions."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

CHOICES = {'open': ('0', '1'), 'closed': ('1', '1'), 'uncertain': ('', '0')}


class ReviewSession:
    def __init__(self, queue, reviewer, split):
        self.queue = Path(queue).resolve()
        if not reviewer.strip() or split not in ('calibration', 'validation', 'test'):
            raise ValueError('A reviewer ID and explicit participant partition are required')
        self.reviewer, self.split = reviewer, split
        self.digest = hashlib.sha256(self.queue.read_bytes()).hexdigest()
        with self.queue.open(newline='', encoding='utf-8-sig') as handle:
            reader = csv.DictReader(handle)
            self.fields = reader.fieldnames
            self.rows = list(reader)
        if not self.rows or not {'sample_id', 'eye_side', 'frame_path', 'subject_id',
                'session_id', 'video_id', 'timestamp_s', 'label', 'observable', 'split',
                'annotation_status', 'orientation_status', 'label_source'} <= set(self.fields or []):
            raise ValueError('Incomplete annotation queue')
        if len({r['subject_id'] for r in self.rows}) != 1:
            raise ValueError('Review one confirmed participant at a time')
        self.groups = {}
        for row in self.rows:
            if row['label'] or row['observable'] or row['split'] or row['label_source']:
                raise ValueError('Input must be the original blank queue')
            path = (self.queue.parent / row['frame_path']).resolve()
            if self.queue.parent not in path.parents or not path.is_file():
                raise ValueError('Review image is missing or outside the session folder')
            group = self.groups.setdefault(row['sample_id'], {})
            if row['eye_side'] not in ('left', 'right') or row['eye_side'] in group:
                raise ValueError('Duplicate or invalid eye side')
            group[row['eye_side']] = row
        for group in self.groups.values():
            if set(group) != {'left', 'right'}:
                raise ValueError('Both anatomical eye rows are required')
            for key in ('frame_path', 'timestamp_s', 'subject_id', 'session_id', 'video_id'):
                if group['left'][key] != group['right'][key]:
                    raise ValueError('Paired eye rows disagree on frame identity')
        self.samples = list(self.groups)
        self.progress_path = self.queue.with_name('human_review_progress.json')
        self.journal_path = self.queue.with_name('human_review_actions.jsonl')
        self.output = self.queue.with_name('human_reviewed.csv')
        if self.output.exists():
            raise FileExistsError('Completed review already exists; preserve it for comparison')
        self.answers = {}
        self.orientation_verified = False
        if self.progress_path.exists():
            saved = json.loads(self.progress_path.read_text(encoding='utf-8'))
            if (saved['queue_sha256'], saved['reviewer_id'], saved['split']) != (
                    self.digest, reviewer, split):
                raise ValueError('Saved review belongs to another queue, reviewer or partition')
            self.answers = saved['answers']
            self.orientation_verified = saved['orientation_verified']
            for sample, answer in self.answers.items():
                if sample not in self.groups or any(answer[side] not in CHOICES for side in ('left', 'right')):
                    raise ValueError('Invalid saved review decision')

    def save(self, sample, left, right, orientation_verified):
        if sample not in self.groups or left not in CHOICES or right not in CHOICES:
            raise ValueError('Choose a state for each eye')
        if orientation_verified is not True:
            raise ValueError('Verify anatomical eye orientation before saving labels')
        answer = dict(left=left, right=right, reviewed_at_utc=datetime.now(timezone.utc).isoformat())
        self.answers[sample] = answer
        self.orientation_verified = True
        with self.journal_path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(dict(sample_id=sample, reviewer_id=self.reviewer,
                queue_sha256=self.digest, split=self.split, **answer)) + '\n')
        draft = dict(schema=1, queue_sha256=self.digest, reviewer_id=self.reviewer,
                     split=self.split, orientation_verified=True, answers=self.answers,
                     predictions_shown=False, completed_frames=len(self.answers), total_frames=len(self.samples))
        temporary = self.progress_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(draft, indent=2), encoding='utf-8')
        temporary.replace(self.progress_path)

    def export(self):
        if set(self.answers) != set(self.samples) or not self.orientation_verified:
            raise ValueError('Review every frame and verify orientation before exporting')
        fields = list(self.fields) + ['reviewer_id', 'reviewed_at_utc']
        with self.output.open('x', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in self.rows:
                answer = self.answers[row['sample_id']]
                label, observable = CHOICES[answer[row['eye_side']]]
                writer.writerow(dict(row, label=label, observable=observable, split=self.split,
                    label_source='human', annotation_status='reviewed', orientation_status='verified',
                    reviewer_id=self.reviewer, reviewed_at_utc=answer['reviewed_at_utc']))
        return self.output


def show_review(session):
    import tkinter as tk
    from tkinter import ttk, messagebox
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title('MARVEL human eye review')
    height = min(810, root.winfo_screenheight()-100)
    root.geometry(f'{min(1050, root.winfo_screenwidth()-80)}x{height}')
    root.minsize(850, 620)
    index = [next((i for i, sample in enumerate(session.samples) if sample not in session.answers), 0)]
    status = tk.StringVar()
    left, right = tk.StringVar(), tk.StringVar()
    orientation = tk.BooleanVar(value=session.orientation_verified)
    ttk.Label(root, text='Review the visible eyes. No CNN, EAR or prompt labels are shown.',
              font=('Segoe UI', 13, 'bold')).pack(pady=(8, 2))
    ttk.Label(root, textvariable=status).pack()
    picture = ttk.Label(root)
    picture.pack(pady=6)
    ttk.Label(root, text='Click the image for a 2x inspection view. Uncertain includes occluded or ambiguous eyes.').pack()
    ttk.Checkbutton(root, text='I verified anatomical left/right orientation for this recording',
                    variable=orientation).pack(pady=5)
    controls = ttk.Frame(root)
    controls.pack()
    for col, (side, value) in enumerate((('Participant LEFT eye', left), ('Participant RIGHT eye', right))):
        ttk.Label(controls, text=side).grid(row=0, column=col, padx=18)
        ttk.Combobox(controls, textvariable=value, values=list(CHOICES), state='readonly', width=20).grid(row=1, column=col, padx=18)

    def refresh():
        sample = session.samples[index[0]]
        row = session.groups[sample]['left']
        source = Image.open(session.queue.parent / row['frame_path'])
        source.load()
        picture.source = source
        preview = source.copy()
        preview.thumbnail((800, max(260, min(400, height-340))))
        picture.photo = ImageTk.PhotoImage(preview)
        picture.configure(image=picture.photo)
        answer = session.answers.get(sample, {})
        left.set(answer.get('left', ''))
        right.set(answer.get('right', ''))
        status.set(f"Frame {index[0]+1}/{len(session.samples)} | {float(row['timestamp_s']):.2f}s | "
                   f"Reviewed {len(session.answers)}/{len(session.samples)} | Partition: {session.split}")

    def move(amount):
        index[0] = max(0, min(len(session.samples)-1, index[0]+amount))
        refresh()

    def save(both=None):
        if both is not None:
            left.set(both)
            right.set(both)
        try:
            session.save(session.samples[index[0]], left.get(), right.get(), orientation.get())
        except ValueError as error:
            messagebox.showerror('Review required', str(error))
            return
        move(1)

    def export():
        try:
            path = session.export()
        except (ValueError, OSError) as error:
            messagebox.showerror('Export not completed', str(error))
            return
        messagebox.showinfo('Review saved', f'Saved {len(session.rows)} reviewed eye rows.\n{path}')
        root.destroy()

    def zoom(event=None):
        window = tk.Toplevel(root)
        window.title('Eye inspection: original frame enlarged')
        canvas = tk.Canvas(window, width=1000, height=650)
        xbar = ttk.Scrollbar(window, orient='horizontal', command=canvas.xview)
        ybar = ttk.Scrollbar(window, orient='vertical', command=canvas.yview)
        canvas.configure(xscrollcommand=xbar.set, yscrollcommand=ybar.set)
        xbar.pack(side='bottom', fill='x'); ybar.pack(side='right', fill='y')
        canvas.pack(fill='both', expand=True)
        image = picture.source.resize((picture.source.width*2, picture.source.height*2))
        canvas.photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor='nw', image=canvas.photo)
        canvas.configure(scrollregion=(0, 0, image.width, image.height))
    picture.bind('<Button-1>', zoom)
    actions = ttk.Frame(root)
    actions.pack(pady=10)
    for column, (text, command) in enumerate((('Both open (O)', lambda: save('open')),
            ('Both closed (C)', lambda: save('closed')), ('Both uncertain (U)', lambda: save('uncertain')),
            ('Save separate eyes', save))):
        ttk.Button(actions, text=text, command=command).grid(row=0, column=column, padx=5)
    nav = ttk.Frame(root)
    nav.pack(pady=4)
    ttk.Button(nav, text='Previous', command=lambda: move(-1)).pack(side='left', padx=5)
    ttk.Button(nav, text='Skip for now', command=lambda: move(1)).pack(side='left', padx=5)
    ttk.Button(nav, text='Export when all frames reviewed', command=export).pack(side='left', padx=5)
    ttk.Label(root, text='Each saved decision is recorded locally. Closing the window preserves saved progress.').pack(pady=6)
    for key, choice in [('o', 'open'), ('c', 'closed'), ('u', 'uncertain')]:
        root.bind(key, lambda event, choice=choice: save(choice))
    refresh()
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    parser.add_argument('--reviewer-id', required=True)
    parser.add_argument('--split', choices=('calibration', 'validation', 'test'), required=True)
    args = parser.parse_args()
    show_review(ReviewSession(args.queue, args.reviewer_id, args.split))


if __name__ == '__main__':
    main()
