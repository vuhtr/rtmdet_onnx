/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class mmdeploy_Classifier */

#ifndef _Included_mmdeploy_Classifier
#define _Included_mmdeploy_Classifier
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:     mmdeploy_Classifier
 * Method:    create
 * Signature: (Ljava/lang/String;Ljava/lang/String;I)J
 */
JNIEXPORT jlong JNICALL Java_mmdeploy_Classifier_create(JNIEnv *, jobject, jstring, jstring, jint);

/*
 * Class:     mmdeploy_Classifier
 * Method:    destroy
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_mmdeploy_Classifier_destroy(JNIEnv *, jobject, jlong);

/*
 * Class:     mmdeploy_Classifier
 * Method:    apply
 * Signature: (J[Lmmdeploy/Mat;[I)[Lmmdeploy/Classifier/Result;
 */
JNIEXPORT jobjectArray JNICALL Java_mmdeploy_Classifier_apply(JNIEnv *, jobject, jlong,
                                                              jobjectArray, jintArray);

#ifdef __cplusplus
}
#endif
#endif