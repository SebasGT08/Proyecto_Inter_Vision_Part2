#include <jni.h>
#include <string>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// Declaración de los clasificadores en cascada
CascadeClassifier clasificadorRostros;
CascadeClassifier clasificadorOjos;
CascadeClassifier clasificadorNariz;
CascadeClassifier clasificadorBoca;

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
double escalaNariz = 1.1;
int minVecinosNariz = 3;
Size minTamanoNariz(20, 20);

// Variables para los parámetros de detección de boca
double escalaBoca = 1.1;
int minVecinosBoca = 5;
Size minTamanoBoca(30, 30);

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

// Función para inicializar los clasificadores en cascada
extern "C" JNIEXPORT void JNICALL
Java_com_example_proyecto_1vison_MainActivity_inicializarCascade(
        JNIEnv* env, jobject, jstring rutaCascadeRostros, jstring rutaCascadeOjos, jstring rutaCascadeNariz, jstring rutaCascadeBoca) {
    const char* rutaRostros = env->GetStringUTFChars(rutaCascadeRostros, 0);
    const char* rutaOjos = env->GetStringUTFChars(rutaCascadeOjos, 0);
    const char* rutaNariz = env->GetStringUTFChars(rutaCascadeNariz, 0);
    const char* rutaBoca = env->GetStringUTFChars(rutaCascadeBoca, 0);

    if (!clasificadorRostros.load(rutaRostros)) {
        std::cerr << "Error cargando el archivo de cascada de rostros" << std::endl;
    }
    if (!clasificadorOjos.load(rutaOjos)) {
        std::cerr << "Error cargando el archivo de cascada de ojos" << std::endl;
    }
    if (!clasificadorNariz.load(rutaNariz)) {
        std::cerr << "Error cargando el archivo de cascada de nariz" << std::endl;
    }
    if (!clasificadorBoca.load(rutaBoca)) {
        std::cerr << "Error cargando el archivo de cascada de boca" << std::endl;
    }

    env->ReleaseStringUTFChars(rutaCascadeRostros, rutaRostros);
    env->ReleaseStringUTFChars(rutaCascadeOjos, rutaOjos);
    env->ReleaseStringUTFChars(rutaCascadeNariz, rutaNariz);
    env->ReleaseStringUTFChars(rutaCascadeBoca, rutaBoca);
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
                                for (const auto& boca : bocas) {
                                    Point centroBoca(rostros[i].x + boca.x + boca.width / 2,
                                                     rostros[i].y + boca.y + boca.height / 2);
                                    if (centroBoca.y > (rostros[i].y + narizSeleccionada.y + narizSeleccionada.height)) {
                                        bocaSeleccionada = boca;
                                        break;
                                    }
                                }

                                if (!bocaSeleccionada.empty()) {
                                    if (modoDibujarGafas) {
                                        // Dibujar las gafas
                                        dibujarGafas(frame, centroOjoIzquierdo, centroOjoDerecho);
                                    } else {
                                        // Dibujar contornos de los ojos, nariz y boca
                                        for (size_t j = 0; j < ojos.size(); j++) {
                                            Point centroOjo(rostros[i].x + ojos[j].x + ojos[j].width / 2,
                                                            rostros[i].y + ojos[j].y + ojos[j].height / 2);
                                            int radio = cvRound((ojos[j].width + ojos[j].height) * 0.25);
                                            circle(frame, centroOjo, radio,
                                                   Scalar(255, 0, 0), 4);
                                        }
                                        Point centroNariz(rostros[i].x + narizSeleccionada.x + narizSeleccionada.width / 2,
                                                          rostros[i].y + narizSeleccionada.y + narizSeleccionada.height / 2);
                                        ellipse(frame, centroNariz, Size(narizSeleccionada.width / 2,
                                                                         narizSeleccionada.height / 2), 0, 0,
                                                360, Scalar(0, 255, 0), 4);

                                        Point centroBoca(rostros[i].x + bocaSeleccionada.x + bocaSeleccionada.width / 2,
                                                         rostros[i].y + bocaSeleccionada.y + bocaSeleccionada.height / 2);
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
