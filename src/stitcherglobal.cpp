#include <string>

int vendor = 1;
int camSrcWidth = 1920; // Default changed to 1920
int camSrcHeight = 1080; // Default changed to 1080

int distorWidth =  1920;
int distorHeight = 1080;

int undistorWidth =  1920; 
int undistorHeight = 1080; 

int stitcherinputWidth = 1920;
int stitcherinputHeight = 1080;

int renderWidth = 1920;
int renderHeight = 1080;
int renderX = 0;
int renderY = 0;
int renderMode = 0;

// output render buffer, in general it's fixed
int renderBufWidth = 1920; 
int renderBufHeight = 1080;

// [Modification]: Set used camera number to 4
int USED_CAMERA_NUM = 4;

bool undistor = true; // Default enable undistortion since we want it by default now

float stitcherMatchConf = 0.3;
float stitcherAdjusterConf = 0.7f;
float stitcherBlenderStrength = 3;
float stitcherCameraExThres = 30;
float stitcherCameraInThres = 100;

int batchSize = 1;
int initMode = 1;