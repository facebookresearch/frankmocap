//Copyright (c) Facebook, Inc. and its affiliates.

#version 140

out vec4 FragColor;
in vec3 Color;

void main() 
{
    FragColor = vec4(Color,1.0);
}
