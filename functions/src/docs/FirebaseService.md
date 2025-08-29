# FirebaseService Class Documentation

## Overview

The `FirebaseService` class is a comprehensive service that provides Firebase Cloud Storage functionality within the Bitcoin trading system. It implements the Singleton design pattern to ensure a single instance manages all Firebase storage operations, providing a unified interface for model weight storage, retrieval, and management.

## Architecture

The FirebaseService implements a singleton pattern with lazy initialization for Firebase Cloud Storage:

```
FirebaseService (Singleton) → Firebase Admin SDK → Google Cloud Storage → Storage Bucket
         ↓                           ↓                    ↓                    ↓
   Single Instance            Authentication         Cloud Storage        File Operations
   Lazy Initialization       Configuration         Bucket Access        Weight Management
```

### Key Design Principles

- **Singleton Pattern**: Ensures single instance across the application
- **Lazy Initialization**: Firebase app initialized only when first accessed
- **Centralized Configuration**: Single point for Firebase configuration management
- **Resource Management**: Efficient bucket access and connection pooling
- **Service Integration**: Seamless integration with other system components

## Class Structure

```typescript
export class FirebaseService {
  private static instance: FirebaseService;
  private bucket: Bucket;

  private constructor();
  public static getInstance(): FirebaseService;
  public getBucket(): Bucket;
}
```

### Core Dependencies

- **firebase-admin**: Firebase Admin SDK for server-side operations
- **@google-cloud/storage**: Google Cloud Storage client library
- **dotenv**: Environment variable management
- **serviceAccount.json**: Firebase service account credentials

## Configuration

### Constructor

The service uses a private constructor with automatic Firebase initialization:

```typescript
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
```

**Configuration Parameters**:

- **Service Account**: Loaded from `serviceAccount.json` file
- **Storage Bucket**: Retrieved from `STORAGE_BUCKET` environment variable
- **Auto-Initialization**: Firebase app created if none exists

### Environment Variables

```bash
STORAGE_BUCKET=your-project-id.appspot.com
```

**Required Configuration**:

- `STORAGE_BUCKET`: Firebase Cloud Storage bucket name
- `serviceAccount.json`: Firebase service account credentials file

## Core Methods

### Public Methods

#### `getInstance(): FirebaseService`

Returns the singleton instance of FirebaseService, creating it if it doesn't exist.

**Returns**:

- `FirebaseService` - The singleton instance

**Usage**:

```typescript
const firebaseService = FirebaseService.getInstance();
```

**Implementation**:

```typescript
public static getInstance(): FirebaseService {
  if (!FirebaseService.instance) {
    FirebaseService.instance = new FirebaseService();
  }
  return FirebaseService.instance;
}
```

#### `getBucket(): Bucket`

Returns the Google Cloud Storage bucket instance for file operations.

**Returns**:

- `Bucket` - Google Cloud Storage bucket instance

**Usage**:

```typescript
const bucket = firebaseService.getBucket();
const file = bucket.file("model-weights.json");
```

**Implementation**:

```typescript
public getBucket(): Bucket {
  return this.bucket;
}
```

## Singleton Pattern Implementation

### Instance Management

The service ensures only one instance exists throughout the application lifecycle:

```typescript
export class FirebaseService {
  private static instance: FirebaseService;

  private constructor() {
    // Private constructor prevents direct instantiation
  }

  public static getInstance(): FirebaseService {
    if (!FirebaseService.instance) {
      FirebaseService.instance = new FirebaseService();
    }
    return FirebaseService.instance;
  }
}
```

**Benefits**:

- **Resource Efficiency**: Single Firebase connection across the application
- **Configuration Consistency**: Unified Firebase configuration
- **Memory Optimization**: Prevents multiple Firebase app instances
- **State Management**: Centralized storage bucket access

### Thread Safety

The singleton implementation is thread-safe for Node.js applications:

- **Single-threaded**: Node.js event loop ensures single execution context
- **Lazy Loading**: Instance created only when first requested
- **Immutable Reference**: Instance reference remains constant after creation

## Firebase Integration

### Admin SDK Initialization

The service automatically initializes Firebase Admin SDK when first accessed:

```typescript
if (!admin.apps.length) {
  admin.initializeApp({
    credential: admin.credential.cert(require("../../../serviceAccount.json")),
    storageBucket: process.env.STORAGE_BUCKET,
  });
}
```

**Initialization Logic**:

- **Conditional Creation**: Only creates Firebase app if none exists
- **Credential Loading**: Loads service account from JSON file
- **Bucket Configuration**: Sets storage bucket from environment variable
- **Automatic Setup**: No manual initialization required

### Storage Bucket Access

Provides direct access to Google Cloud Storage bucket operations:

```typescript
this.bucket = admin.storage().bucket();
```

## File Operations

### Model Weight Storage

The service is primarily used for storing and retrieving model weights:

```typescript
// Store model weights
const bucket = FirebaseService.getInstance().getBucket();
const file = bucket.file("tradePredictorWeights.json");

await file.save(JSON.stringify(modelWeights), {
  metadata: {
    contentType: "application/json",
    metadata: {
      modelVersion: "1.0.0",
      timestamp: new Date().toISOString(),
    },
  },
});
```

### Model Weight Retrieval

```typescript
// Retrieve model weights
const bucket = FirebaseService.getInstance().getBucket();
const file = bucket.file("tradePredictorWeights.json");

const [exists] = await file.exists();
if (exists) {
  const [data] = await file.download();
  const modelWeights = JSON.parse(data.toString());
  return modelWeights;
}
```

## Integration Examples

### Trading Model Integration

```typescript
// Integration with trading model
class TradingModel {
  private firebaseService: FirebaseService;

  constructor() {
    this.firebaseService = FirebaseService.getInstance();
  }

  async saveModelWeights(weights: any, version: string = "1.0.0") {
    const bucket = this.firebaseService.getBucket();
    const file = bucket.file("tradePredictorWeights.json");

    const weightData = {
      weights,
      metadata: {
        version,
        timestamp: new Date().toISOString(),
        modelType: "bitcoin-trading",
        features: 26,
        architecture: "Conv1D-LSTM-Dense",
      },
    };

    await file.save(JSON.stringify(weightData), {
      metadata: {
        contentType: "application/json",
        metadata: {
          modelVersion: version,
          timestamp: new Date().toISOString(),
        },
      },
    });

    console.log(`Model weights saved successfully: ${version}`);
  }

  async loadModelWeights(): Promise<any> {
    const bucket = this.firebaseService.getBucket();
    const file = bucket.file("tradePredictorWeights.json");

    try {
      const [exists] = await file.exists();
      if (!exists) {
        throw new Error("Model weights file not found");
      }

      const [data] = await file.download();
      const weightData = JSON.parse(data.toString());

      console.log(`Model weights loaded: ${weightData.metadata.version}`);
      return weightData.weights;
    } catch (error) {
      console.error("Failed to load model weights:", error);
      throw error;
    }
  }
}
```

### Backup and Versioning

```typescript
// Model versioning and backup
class ModelVersionManager {
  private firebaseService: FirebaseService;

  constructor() {
    this.firebaseService = FirebaseService.getInstance();
  }

  async createBackup(weights: any, version: string) {
    const bucket = this.firebaseService.getBucket();
    const backupFile = bucket.file(`backups/model-weights-${version}.json`);

    const backupData = {
      weights,
      metadata: {
        version,
        timestamp: new Date().toISOString(),
        backupType: "manual",
        originalFile: "tradePredictorWeights.json",
      },
    };

    await backupFile.save(JSON.stringify(backupData), {
      metadata: {
        contentType: "application/json",
        metadata: {
          version,
          timestamp: new Date().toISOString(),
        },
      },
    });

    console.log(`Backup created: model-weights-${version}.json`);
  }

  async listBackups(): Promise<string[]> {
    const bucket = this.firebaseService.getBucket();
    const [files] = await bucket.getFiles({ prefix: "backups/" });

    return files.map((file) => file.name);
  }

  async restoreBackup(version: string): Promise<any> {
    const bucket = this.firebaseService.getBucket();
    const backupFile = bucket.file(`backups/model-weights-${version}.json`);

    const [data] = await backupFile.download();
    const backupData = JSON.parse(data.toString());

    // Restore to main weights file
    const mainFile = bucket.file("tradePredictorWeights.json");
    await mainFile.save(JSON.stringify(backupData), {
      metadata: {
        contentType: "application/json",
        metadata: {
          modelVersion: version,
          timestamp: new Date().toISOString(),
          restoredFrom: `backups/model-weights-${version}.json`,
        },
      },
    });

    console.log(`Backup restored: ${version}`);
    return backupData.weights;
  }
}
```

## Error Handling

### Comprehensive Error Management

The service provides robust error handling for various failure scenarios:

```typescript
class FirebaseServiceError extends Error {
  constructor(message: string, public code: string, public details?: any) {
    super(message);
    this.name = "FirebaseServiceError";
  }
}

// Error handling example
async function safeFileOperation(operation: () => Promise<any>) {
  try {
    return await operation();
  } catch (error: any) {
    if (error.code === "storage/unauthorized") {
      throw new FirebaseServiceError(
        "Firebase authentication failed",
        "AUTH_ERROR",
        error
      );
    } else if (error.code === "storage/not-found") {
      throw new FirebaseServiceError(
        "File not found in storage",
        "FILE_NOT_FOUND",
        error
      );
    } else {
      throw new FirebaseServiceError(
        "Firebase storage operation failed",
        "STORAGE_ERROR",
        error
      );
    }
  }
}
```

### Common Error Scenarios

1. **Authentication Errors**: Invalid service account credentials
2. **Permission Errors**: Insufficient storage bucket permissions
3. **Network Errors**: Connection timeouts or network failures
4. **File Not Found**: Attempting to access non-existent files
5. **Quota Exceeded**: Storage quota or rate limit exceeded

## Performance Considerations

### Connection Pooling

The singleton pattern ensures efficient connection management:

```typescript
// Efficient connection reuse
const firebaseService = FirebaseService.getInstance();
const bucket = firebaseService.getBucket();

// Multiple operations use the same connection
const file1 = bucket.file("file1.json");
const file2 = bucket.file("file2.json");
```

### Caching Strategy

For frequently accessed data, consider implementing caching:

```typescript
class CachedFirebaseService {
  private cache = new Map<string, { data: any; timestamp: number }>();
  private CACHE_DURATION = 5 * 60 * 1000; // 5 minutes

  async getCachedFile(filePath: string): Promise<any> {
    const cached = this.cache.get(filePath);

    if (cached && Date.now() - cached.timestamp < this.CACHE_DURATION) {
      return cached.data;
    }

    const firebaseService = FirebaseService.getInstance();
    const bucket = firebaseService.getBucket();
    const file = bucket.file(filePath);

    const [data] = await file.download();
    const parsedData = JSON.parse(data.toString());

    this.cache.set(filePath, { data: parsedData, timestamp: Date.now() });
    return parsedData;
  }
}
```

## Security Considerations

### Service Account Security

1. **Credential Protection**: Store service account JSON securely
2. **Access Control**: Use minimal required permissions
3. **Environment Isolation**: Separate credentials for different environments
4. **Regular Rotation**: Periodically rotate service account keys

### Data Security

1. **Encryption**: Enable server-side encryption for sensitive data
2. **Access Logging**: Monitor file access patterns
3. **Version Control**: Maintain audit trail of file changes
4. **Backup Security**: Secure backup storage locations

## Testing

### Unit Testing

```typescript
// Example unit test
describe("FirebaseService", () => {
  let firebaseService: FirebaseService;

  beforeEach(() => {
    // Mock Firebase Admin SDK
    jest.mock("firebase-admin", () => ({
      apps: [],
      initializeApp: jest.fn(),
      storage: jest.fn(() => ({
        bucket: jest.fn(() => ({
          file: jest.fn(() => ({
            save: jest.fn(),
            download: jest.fn(),
            exists: jest.fn(),
          })),
        })),
      })),
    }));

    firebaseService = FirebaseService.getInstance();
  });

  it("should return singleton instance", () => {
    const instance1 = FirebaseService.getInstance();
    const instance2 = FirebaseService.getInstance();
    expect(instance1).toBe(instance2);
  });

  it("should provide bucket access", () => {
    const bucket = firebaseService.getBucket();
    expect(bucket).toBeDefined();
  });
});
```

### Integration Testing

```typescript
// Integration test with real Firebase
describe("FirebaseService Integration", () => {
  it("should save and retrieve model weights", async () => {
    const firebaseService = FirebaseService.getInstance();
    const bucket = firebaseService.getBucket();

    const testWeights = { layer1: [1, 2, 3], layer2: [4, 5, 6] };
    const file = bucket.file("test-weights.json");

    // Save weights
    await file.save(JSON.stringify(testWeights));

    // Retrieve weights
    const [data] = await file.download();
    const retrievedWeights = JSON.parse(data.toString());

    expect(retrievedWeights).toEqual(testWeights);

    // Cleanup
    await file.delete();
  });
});
```

## Monitoring and Logging

### Performance Monitoring

```typescript
// Performance monitoring
class FirebaseServiceMonitor {
  private metrics = {
    readOperations: 0,
    writeOperations: 0,
    errors: 0,
    totalLatency: 0,
  };

  async monitorOperation<T>(operation: () => Promise<T>): Promise<T> {
    const startTime = Date.now();

    try {
      const result = await operation();
      const latency = Date.now() - startTime;

      this.metrics.totalLatency += latency;
      console.log(`Operation completed in ${latency}ms`);

      return result;
    } catch (error) {
      this.metrics.errors++;
      console.error("Operation failed:", error);
      throw error;
    }
  }

  getMetrics() {
    return {
      ...this.metrics,
      averageLatency:
        this.metrics.totalLatency /
        (this.metrics.readOperations + this.metrics.writeOperations),
    };
  }
}
```

### Logging Strategy

```typescript
// Structured logging
const logger = {
  info: (message: string, metadata?: any) => {
    console.log(
      JSON.stringify({
        level: "info",
        message,
        timestamp: new Date().toISOString(),
        service: "FirebaseService",
        ...metadata,
      })
    );
  },

  error: (message: string, error?: any) => {
    console.error(
      JSON.stringify({
        level: "error",
        message,
        timestamp: new Date().toISOString(),
        service: "FirebaseService",
        error: error?.message || error,
      })
    );
  },
};
```

## Best Practices

### Configuration Management

1. **Environment Variables**: Use environment variables for configuration
2. **Service Account**: Store service account JSON securely
3. **Bucket Naming**: Use consistent bucket naming conventions
4. **File Organization**: Organize files with clear directory structure

### Error Handling

1. **Graceful Degradation**: Handle errors without crashing the application
2. **Retry Logic**: Implement exponential backoff for transient errors
3. **Error Logging**: Log errors with sufficient context for debugging
4. **Fallback Mechanisms**: Provide fallback options when Firebase is unavailable

### Performance Optimization

1. **Connection Reuse**: Leverage singleton pattern for connection efficiency
2. **Batch Operations**: Group multiple operations when possible
3. **Caching**: Implement appropriate caching for frequently accessed data
4. **Async Operations**: Use async/await for non-blocking operations

## Troubleshooting

### Common Issues

1. **Authentication Failures**: Verify service account credentials
2. **Permission Errors**: Check bucket permissions and IAM roles
3. **Network Timeouts**: Verify network connectivity and firewall settings
4. **Quota Exceeded**: Monitor storage usage and implement cleanup

### Debug Mode

Enable debug logging for troubleshooting:

```typescript
// Debug configuration
const DEBUG_MODE = process.env.FIREBASE_DEBUG === "true";

if (DEBUG_MODE) {
  console.log("Firebase Service Debug Mode Enabled");
  console.log("Storage Bucket:", process.env.STORAGE_BUCKET);
  console.log(
    "Service Account Path:",
    require.resolve("../../../serviceAccount.json")
  );
}
```

## Conclusion

The FirebaseService provides a robust, efficient interface for Firebase Cloud Storage operations within the Bitcoin trading system. Its singleton pattern ensures optimal resource utilization while providing comprehensive error handling and performance monitoring capabilities.

The service seamlessly integrates with the trading model architecture, enabling reliable storage and retrieval of model weights, backup management, and version control. Whether used for production model deployment or development testing, the FirebaseService delivers consistent, secure, and performant storage operations.

The comprehensive error handling, security considerations, and monitoring capabilities make it an essential component for maintaining reliable model weight management in the trading system.
