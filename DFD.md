# Data Flow Diagram (DFD) - AI-Based Crowd Panic Prediction System

This document contains the Data Flow Diagrams for the Crowd Panic Prediction System. You can view these diagrams by opening this Markdown file in an editor that supports Mermaid.js (such as VS Code with a Markdown Preview extension, Obsidian, or GitHub).

## Level 0 Context Diagram
This diagram shows the system as a single process interacting with the Client.

```mermaid
flowchart LR
    Client([Client Application / User])
    System((0. Crowd Panic\nPrediction System))
    
    Client -- "HTTP POST (Image/Video)" --> System
    System -- "JSON Response\n(Scores, Risk Level, Annotated Image)" --> Client
```

---

## Level 1 System Data Flow Diagram
This details the internal flow of data between the FastAPI endpoints and the various Python processing modules (`detection`, `density`, `movement`, `risk`).

```mermaid
flowchart TD
    Client([Client Application])
    
    subgraph System Boundary: Crowd Panic Prediction API
        P1((1. Request\nHandler\nmain.py))
        P2((2. Image\nDecoder\nutils.py))
        P3((3. Person\nDetection\ndetection.py))
        P4((4. Density\nCalculation\ndensity.py))
        P5((5. Movement\nAnalysis\nmovement.py))
        P6((6. Risk\nClassification\nrisk.py))
        
        P1 -- "Raw Image/Video Bytes" --> P2
        P2 -- "Decoded Frames\n(NumPy Arrays)" --> P1
        
        P1 -- "Decoded Frame" --> P3
        P3 -- "Bounding Boxes,\nPeople Count" --> P1
        
        P1 -- "Bounding Boxes,\nFrame Dimensions" --> P4
        P4 -- "Density Score\n(0.0 - 1.0)" --> P1
        
        P1 -- "Decoded Frame(s)" --> P5
        P5 -- "Movement Score\n(0.0 - 1.0)" --> P1
        
        P1 -- "Density Score,\nMovement Score" --> P6
        P6 -- "Risk Level\n(Green/Yellow/Red)" --> P1
    end
    
    Client -- "POST /analyze\nmultipart/form-data" --> P1
    P1 -- "JSON Response\n(Risk assessment)" --> Client
```
