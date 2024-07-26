#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <android/log.h>

using namespace cv;
using namespace std;

#define LOG_TAG "native-lib"
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// Declaración de los clasificadores en cascada
CascadeClassifier clasificadorRostros;
CascadeClassifier clasificadorOjos;
CascadeClassifier clasificadorNariz;
CascadeClassifier clasificadorBoca;

//Ruta del asset del bigote
string rutaBigote;

// Variables para los parámetros de detección de rostros
double escalaRostros = 1.1;
int minVecinosRostros = 5;
int flagsRostros = 0 | CASCADE_SCALE_IMAGE;
Size minTamanoRostros(30, 30);

// Variables para los parámetros de detección de ojos
double escalaOjos = 1.1;
int minVecinosOjos = 15;
int flagsOjos = 0 | CASCADE_SCALE_IMAGE;
Size minTamanoOjos(20, 20);


// Variables para los parámetros de detección de nariz
double escalaNariz = 1.5;
int minVecinosNariz = 3;
Size minTamanoNariz(20, 20);

// Variables para los parámetros de detección de boca
double escalaBoca = 1.5;
int minVecinosBoca = 5;
Size minTamanoBoca(20, 20);

// Función para dibujar gafas en la imagen
void dibujarGafas(Mat& imagen, Point centroOjoIzquierdo, Point centroOjoDerecho) {
    const int patronGafas[5][26] = {
            {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
            {1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1},
            {0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0},
            {0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 0},
            {0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0}
    };

    int altoPatron = 5;
    int anchoPatron = 26;
    Scalar color = Scalar(255, 0, 255);

    // Calcular el centro de las gafas
    Point centroGafas = (centroOjoIzquierdo + centroOjoDerecho) * 0.5;

    // Calcular el ancho y la altura de las gafas
    int anchoGafas = norm(centroOjoIzquierdo - centroOjoDerecho) * 2;
    int altoGafas = anchoGafas / 4;

    // Calcular la posición de la esquina superior izquierda de las gafas
    int x = centroGafas.x - anchoGafas / 2;
    int y = centroGafas.y - altoGafas / 2;

    for (int i = 0; i < altoPatron; ++i) {
        for (int j = 0; j < anchoPatron; ++j) {
            if (patronGafas[i][j] == 1) {
                int pixelX = x + j * anchoGafas / anchoPatron;
                int pixelY = y + i * altoGafas / altoPatron;
                if (pixelY >= 0 && pixelY < imagen.rows && pixelX >= 0 && pixelX < imagen.cols) {
                    rectangle(imagen, Point(pixelX, pixelY),
                              Point(pixelX + anchoGafas / anchoPatron,
                                    pixelY + altoGafas / altoPatron), color, FILLED);
                }
            }
        }
    }
}

void dibujarBigote(Mat& imagen, Point centroOjoIzquierdo, Point centroOjoDerecho, Point centroNariz, Point centroBoca) {
    // Cargar la imagen del bigote desde la ruta especificada
    Mat bigote = imread(rutaBigote, IMREAD_UNCHANGED);
    if (bigote.empty()) {
        LOGE("No se pudo cargar la imagen del bigote desde %s", rutaBigote.c_str());
        return;
    }
    // Calcular la distancia entre los ojos para determinar el tamaño del bigote
    int distanciaOjos = norm(centroOjoIzquierdo - centroOjoDerecho);

    // Establecer un tamaño relativo para el bigote basado en la distancia entre los ojos
    int anchoBigote = distanciaOjos * 1.5;  // Ajusta este factor según sea necesario
    int altoBigote = bigote.rows * anchoBigote / bigote.cols;

    // Redimensionar la imagen del bigote
    resize(bigote, bigote, Size(anchoBigote, altoBigote));

    // Calcular la posición superior izquierda del bigote
    int x = centroNariz.x - anchoBigote / 2;
    int y = centroNariz.y + (centroBoca.y - centroNariz.y) / 2 - altoBigote / 2;

    // Crear una región de interés en la imagen original
    Rect roi(x, y, anchoBigote, altoBigote);
    Mat imagenROI = imagen(roi);

    // Combinar la imagen del bigote con la imagen original
    for (int i = 0; i < bigote.rows; ++i) {
        for (int j = 0; j < bigote.cols; ++j) {
            Vec4b& bgra = bigote.at<Vec4b>(i, j);
            if (bgra[3] > 0) { // Verificar el canal alfa para la transparencia
                imagenROI.at<Vec3b>(i, j) = Vec3b(bgra[0], bgra[1], bgra[2]);
            }
        }
    }
}


// Función para inicializar los clasificadores en cascada
extern "C" JNIEXPORT void JNICALL
Java_com_example_proyecto_1vison_MainActivity_inicializarCascade(
        JNIEnv* env, jobject, jstring rutaCascadeRostros, jstring rutaCascadeOjos, jstring rutaCascadeNariz, jstring rutaCascadeBoca, jstring rutaBigoteJNI) {
    const char* rutaRostros = env->GetStringUTFChars(rutaCascadeRostros, 0);
    const char* rutaOjos = env->GetStringUTFChars(rutaCascadeOjos, 0);
    const char* rutaNariz = env->GetStringUTFChars(rutaCascadeNariz, 0);
    const char* rutaBoca = env->GetStringUTFChars(rutaCascadeBoca, 0);
    const char* rutaBigoteC = env->GetStringUTFChars(rutaBigoteJNI, 0);

    if (!clasificadorRostros.load(rutaRostros)) {
        LOGE("Error cargando el archivo de cascada de rostros");
    }
    if (!clasificadorOjos.load(rutaOjos)) {
        LOGE("Error cargando el archivo de cascada de ojos");
    }
    if (!clasificadorNariz.load(rutaNariz)) {
        LOGE("Error cargando el archivo de cascada de nariz");
    }
    if (!clasificadorBoca.load(rutaBoca)) {
        LOGE("Error cargando el archivo de cascada de boca");
    }

    rutaBigote = string(rutaBigoteC);

    env->ReleaseStringUTFChars(rutaCascadeRostros, rutaRostros);
    env->ReleaseStringUTFChars(rutaCascadeOjos, rutaOjos);
    env->ReleaseStringUTFChars(rutaCascadeNariz, rutaNariz);
    env->ReleaseStringUTFChars(rutaCascadeBoca, rutaBoca);
    env->ReleaseStringUTFChars(rutaBigoteJNI, rutaBigoteC);

    LOGD("Cascadas y ruta del bigote inicializadas");
}

// Función para detectar rostros y ojos
void detectar(Mat& frame, bool modoDibujarGafas) {
    Mat frameGris;
    cvtColor(frame, frameGris, COLOR_RGBA2GRAY);
    equalizeHist(frameGris, frameGris);

    // Detectar rostros
    std::vector<Rect> rostros;
    clasificadorRostros.detectMultiScale(
            frameGris, rostros, escalaRostros,
            minVecinosRostros, flagsRostros, minTamanoRostros);

    for (size_t i = 0; i < rostros.size(); i++) {
        if (!modoDibujarGafas) {
            Point centro(rostros[i].x + rostros[i].width / 2,
                         rostros[i].y + rostros[i].height / 2);
            ellipse(frame, centro, Size(rostros[i].width / 2,
                                        rostros[i].height / 2), 0, 0,
                    360, Scalar(255, 0, 255), 4);
        }

        Mat rostroROI = frameGris(rostros[i]);

        // En cada rostro, detectar ojos
        std::vector<Rect> ojos;
        clasificadorOjos.detectMultiScale(
                rostroROI, ojos, escalaOjos,
                minVecinosOjos, flagsOjos, minTamanoOjos);

        // Asegurarse de que al menos se detectan 2 ojos para dibujar las gafas o los contornos
        if (ojos.size() >= 2) {
            // Ordenar los ojos por su posición x para asegurar que los ojos más cercanos se dibujan
            sort(ojos.begin(), ojos.end(), [](const Rect& a, const Rect& b) {
                return a.x < b.x;
            });

            // Validar la posición vertical de los ojos
            if (ojos[0].y < rostros[i].height / 2 && ojos[1].y < rostros[i].height / 2) {
                Point centroOjoIzquierdo(rostros[i].x + ojos[0].x + ojos[0].width / 2,
                                         rostros[i].y + ojos[0].y + ojos[0].height / 2);
                Point centroOjoDerecho(rostros[i].x + ojos[1].x + ojos[1].width / 2,
                                       rostros[i].y + ojos[1].y + ojos[1].height / 2);

                // Validar la distancia entre los ojos
                double distanciaOjos = norm(centroOjoIzquierdo - centroOjoDerecho);
                if (distanciaOjos > rostros[i].width * 0.2 && distanciaOjos < rostros[i].width * 0.6) {
                    // Detectar nariz
                    std::vector<Rect> narices;
                    clasificadorNariz.detectMultiScale(
                            rostroROI, narices, escalaNariz,
                            minVecinosNariz, 0, minTamanoNariz);

                    if (!narices.empty()) {
                        Rect narizSeleccionada;
                        //Validar la nariz para usar las mas cercana al centro y que este debajo de los ojos
                        for (const auto& nariz : narices) {
                            Point centroNariz(rostros[i].x + nariz.x + nariz.width / 2,
                                              rostros[i].y + nariz.y + nariz.height / 2);
                            if (centroNariz.y > centroOjoIzquierdo.y && centroNariz.y > centroOjoDerecho.y) {
                                narizSeleccionada = nariz;
                                break;
                            }
                        }

                        if (!narizSeleccionada.empty()) {
                            // Detectar boca
                            std::vector<Rect> bocas;
                            clasificadorBoca.detectMultiScale(
                                    rostroROI, bocas, escalaBoca,
                                    minVecinosBoca, 0, minTamanoBoca);

                            if (!bocas.empty()) {
                                Rect bocaSeleccionada;
                                //Validar la boca y usar la mas cercana al centro de la nariz y debajo de la nariz
                                for (const auto& boca : bocas) {
                                    Point centroBoca(rostros[i].x + boca.x + boca.width / 2,
                                                     rostros[i].y + boca.y + boca.height / 2);
                                    if (centroBoca.y > (rostros[i].y + narizSeleccionada.y + narizSeleccionada.height)) {
                                        bocaSeleccionada = boca;
                                        break;
                                    }
                                }

                                if (!bocaSeleccionada.empty()) {
                                    Point centroNariz(rostros[i].x + narizSeleccionada.x + narizSeleccionada.width / 2,
                                                      rostros[i].y + narizSeleccionada.y + narizSeleccionada.height / 2);
                                    Point centroBoca(rostros[i].x + bocaSeleccionada.x + bocaSeleccionada.width / 2,
                                                     rostros[i].y + bocaSeleccionada.y + bocaSeleccionada.height / 2);

                                    if (modoDibujarGafas) {
                                        // Dibujar las gafas
                                        dibujarGafas(frame, centroOjoIzquierdo, centroOjoDerecho);
                                        // Dibujar el bigote
                                        dibujarBigote(frame, centroOjoIzquierdo, centroOjoDerecho,centroNariz, centroBoca);
                                    } else {
                                        // Dibujar contornos de los ojos, nariz y boca
                                        for (size_t j = 0; j < ojos.size(); j++) {
                                            Point centroOjo(rostros[i].x + ojos[j].x + ojos[j].width / 2,
                                                            rostros[i].y + ojos[j].y + ojos[j].height / 2);
                                            int radio = cvRound((ojos[j].width + ojos[j].height) * 0.25);
                                            circle(frame, centroOjo, radio,
                                                   Scalar(255, 0, 0), 4);
                                        }

                                        ellipse(frame, centroNariz, Size(narizSeleccionada.width / 2,
                                                                         narizSeleccionada.height / 2), 0, 0,
                                                360, Scalar(0, 255, 0), 4);

                                        ellipse(frame, centroBoca, Size(bocaSeleccionada.width / 2,
                                                                        bocaSeleccionada.height / 2), 0, 0,
                                                360, Scalar(0, 0, 255), 4);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}


// Función para procesar cada frame, detectar y dibujar
extern "C" JNIEXPORT void JNICALL
Java_com_example_proyecto_1vison_MainActivity_procesarFrame(
        JNIEnv* env, jobject, jlong direccionMatRgba, jboolean modoDibujarGafas) {
    Mat& frame = *(Mat*)direccionMatRgba;
    Mat frameBgr;
    cvtColor(frame, frameBgr, COLOR_RGBA2BGR); // Convertir a BGR para el procesamiento
    detectar(frameBgr, modoDibujarGafas);
    cvtColor(frameBgr, frame, COLOR_BGR2RGBA); // Convertir de nuevo a RGBA
}
