//Copyright (c) Facebook, Inc. and its affiliates.

#version 140

in vec3 a_Position;
in vec3 a_Normal;
in vec3 a_Color;

out vec3 CamNormal;
out vec3 CamPos;
out vec3 Color;

uniform mat4 ModelMat;
uniform mat4 PerspMat;

void main()
{
	gl_Position = PerspMat * ModelMat * vec4(a_Position, 1.0);
    CamNormal = (ModelMat * vec4(a_Normal, 0.0)).xyz;
    CamPos = (ModelMat * vec4(a_Position, 1.0)).xyz;

    //Color = vec3(1.0, 1.0, 1.0);
    Color = a_Color;
}