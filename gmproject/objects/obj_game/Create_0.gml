randomize();
draw_set_font(fnt_m3x6);
display_set_gui_maximize();
game_set_speed(60, gamespeed_fps);

timeScale = 1;

global.dt = 0;
global.gridSize = 12;
global.solidmap = [[]];
global.trainingMode = false;
global.debugMode = false;
global.playerOverrideMode = false;

#macro DT global.dt
#macro GRIDW global.gridSize
#macro GRIDH global.gridSize
#macro SOLIDMAP global.solidmap
#macro TRAINING_MODE global.trainingMode
#macro DEBUG global.debugMode
#macro PLAYER_OVERRIDE global.playerOverrideMode

toggleTrainingMode = function() { TRAINING_MODE = !TRAINING_MODE; }