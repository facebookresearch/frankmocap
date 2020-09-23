//Copyright (c) Facebook, Inc. and its affiliates.

#version 140

in vec3 a_Position;
in vec3 a_Color;

out vec3 Color;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(a_Position, 1.0);
    Color = a_Color;
}