# README

---
### Project Details

The simulation contains a single agent that navigates a large environment.  At each time step, it has four actions at its disposal:
- `0`: walk forward 
- `1`: walk backward
- `2`: turn left
- `3`: turn right

The state space has `37` dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  A reward of `+1` is provided for collecting a yellow banana, and a reward of `-1` is provided for collecting a blue banana. The task is episodic and the environment is considered solved when the agent gets an average score of `+13` over 100 consecutive episodes.

### Software and Libraries

Please install [Unity ML-Agents](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Installation.md) and make sure that the following packages are installed too:
- pandas
- numpy
- matplotlib
- torch

### Getting Started
```
git clone https://github.com/QuantLandi/navigation-dqn.git
cd navigation
your-markdown-viewer report.md
python run.py
```
To train the agent and evaluate its performance, please open the `navigation.ipynb` notebook and execute all cells.

Please make sure to change the `filename` variable in `run.py` to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Banana.app"`
- **Windows** (x86): `"path/to/Banana_Windows_x86/Banana.exe"`
- **Windows** (x86_64): `"path/to/Banana_Windows_x86_64/Banana.exe"`
- **Linux** (x86): `"path/to/Banana_Linux/Banana.x86"`
- **Linux** (x86_64): `"path/to/Banana_Linux/Banana.x86_64"`
- **Linux** (x86, headless): `"path/to/Banana_Linux_NoVis/Banana.x86"`
- **Linux** (x86_64, headless): `"path/to/Banana_Linux_NoVis/Banana.x86_64"`

For instance, if you are using a Mac, then you downloaded `Banana.app`.  If this file is in the same folder as the notebook, then the line below should appear as follows:
```
env = UnityEnvironment(file_name="Banana.app")
``` 
