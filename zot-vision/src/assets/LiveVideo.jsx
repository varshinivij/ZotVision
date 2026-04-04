import myVideo from './assets/video.mp4';
import { useState } from 'react'

function VideoPlayer(props) {
    processed, setProcessed = useState('false')
    return (
        <div>
            <h1> Live Feed: Camera {props.id} </h1>
            <image src={props.url} />
            {processed && <h2>Processed</h2>}   
            {!processed && <h2>Not Processed</h2>}  
        </div>
    );
}