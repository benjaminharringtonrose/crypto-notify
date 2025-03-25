import * as admin from "firebase-admin";
import dotenv from "dotenv";
import { Bucket } from "@google-cloud/storage";

dotenv.config();

export class FirebaseService {
  private static instance: FirebaseService;
  private bucket: Bucket;

  private constructor() {
    if (!admin.apps.length) {
      admin.initializeApp({
        credential: admin.credential.cert(
          require("../../../serviceAccount.json")
        ),
        storageBucket: process.env.STORAGE_BUCKET,
      });
    }
    this.bucket = admin.storage().bucket();
  }

  public static getInstance(): FirebaseService {
    if (!FirebaseService.instance) {
      FirebaseService.instance = new FirebaseService();
    }
    return FirebaseService.instance;
  }

  public getBucket(): Bucket {
    return this.bucket;
  }
}
