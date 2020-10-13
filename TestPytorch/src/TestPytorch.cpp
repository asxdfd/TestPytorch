// TestPytorch.cpp : Defines the entry point for the application.
//

#include "TestPytorch.h"

using namespace std;

int main()
{
    FaceDetector face("D:\\PycharmProjects\\tf2torch\\faceboxes.pt");
    face.predict("20200323133017663.jpg", "res_faceboxes.jpg");

    return 0;
}
